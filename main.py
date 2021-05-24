import os
import shutil
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AdamW
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt



from utils.functions import load_model, WordSplitTokenizer
from utils.args_helper import get_parser, print_opts, append_dataset_args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

###
# modelling functions
###
def get_lr(args, optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

###
# Training & Evaluation Function
###

# Evaluate function for validation and test
def evaluate(model, data_loader, forward_fn, metrics_fn, i2w, is_test=False, epoch=0):
    model.eval()
    total_loss, total_correct, total_labels = 0, 0, 0
    eval_global_step = 0
    eval_step_loss_histories = []
    steps = []

    list_hyp, list_label, list_seq = [], [], []

    pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))  
    print('test loader len: ',len(data_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]        
        loss, batch_hyp, batch_label = forward_fn(model, batch_data[:-1], i2w=i2w, device=args['device'])

        
        # Calculate total loss
        test_loss = loss.item()
        total_loss = total_loss + test_loss
        print('test loss: ',test_loss)

        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        list_seq += batch_seq
        metrics = metrics_fn(list_hyp, list_label)

        if not is_test:
            pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
        else:
            pbar.set_description("TEST LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))

        eval_global_step+=1
        #save history metrics per save_steps
        #for train loss visualitation
        if eval_global_step % args["save_eval_history_steps"] == 0 :
            print('save eval loss history at global step: ', eval_global_step)
            step_eval_loss = total_loss/(i+1)
            eval_step_loss_histories.append(step_eval_loss)
            steps.append(eval_global_step)

    
    final_eval_loss = total_loss/(i+1)
    if is_test:
        return total_loss, metrics, list_hyp, list_label, list_seq
    else:
        return total_loss, metrics, final_eval_loss

# Training function and trainer
def train(model, train_loader, valid_loader, optimizer, forward_fn, metrics_fn, valid_criterion, i2w, n_epochs, evaluate_every=1, early_stop=3, step_size=1, gamma=0.5, model_dir="", exp_id=None):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_metric = -100
    count_stop = 0
    val_loss_history = []
    train_loss_history = []
    step_train_loss_history = []
    step_val_loss_history = []
    global_step = 0
    train_history_steps = []
    val_history_steps = []

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        list_hyp, list_label = [], []
        
        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            loss, batch_hyp, batch_label = forward_fn(model, batch_data[:-1], i2w=i2w, device=args['device'])

            optimizer.zero_grad()
            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_norm'])
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_norm'])
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label
            
            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), get_lr(args, optimizer)))

            global_step += 1

            #save history metrics per save_steps
            #for train loss visualitation
            if global_step % args["save_history_steps"] == 0 :
                print('save training loss history at global step: ', global_step)
                step_train_loss = total_train_loss/(i+1)
                step_train_loss_history.append(step_train_loss)
                train_history_steps.append(global_step)

                        
        metrics = metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(args, optimizer)))
        
        #save history metrics per epoch
        epoch_train_loss = total_train_loss/(i+1)
        train_loss_history.append(epoch_train_loss)

        # Decay Learning Rate
        scheduler.step()

        # evaluate
        if ((epoch+1) % evaluate_every) == 0:
            val_loss, val_metrics,  eval_loss_per_epoch = evaluate(model, valid_loader, forward_fn, metrics_fn, i2w, is_test=False)

            #for val loss visualitation
            # step_val_loss_history += eval_step_loss_histories
            # val_history_steps += val_steps
            val_loss_history.append(eval_loss_per_epoch)

            # Early stopping
            val_metric = val_metrics[valid_criterion]
            if best_val_metric < val_metric:
                best_val_metric = val_metric
                # save model
                if exp_id is not None:
                    torch.save(model.state_dict(), model_dir + "/best_model_" + str(exp_id) + ".th")
                else:
                    torch.save(model.state_dict(), model_dir + "/best_model.th")
                count_stop = 0
            else:
                count_stop += 1
                print("count stop:", count_stop)
                if count_stop == early_stop:
                    break
    
    #training/validation per step
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.savefig(model_dir + "/visualization/" + 'train_val_loss.png')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    #training plot per step
    plt.plot(train_history_steps,step_train_loss_history)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.legend(['train loss'], loc='upper left')
    plt.savefig(model_dir + "/visualization/" + 'train_per_step_loss.png')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    #validation plot per step
    # plt.plot(val_history_steps,step_val_loss_history)
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('step')
    # plt.legend(['train loss', 'val loss'], loc='upper left')
    # plt.savefig(model_dir + "/visualization/" + 'val_per_step_loss.png')
    # plt.show()
    

if __name__ == "__main__":
    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True
    
    # Parse args
    args = get_parser()
    args = append_dataset_args(args)

    # create directory
    model_dir = '{}/{}/{}'.format(args["model_dir"],args["dataset"],args['experiment_name'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    elif args['force']:
        print(f'overwriting model directory `{model_dir}`')
    else:
        raise Exception(f'model directory `{model_dir}` already exists, use --force if you want to overwrite the folder')

    visualization_dir = '{}/{}'.format(model_dir,'visualization')
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir, exist_ok=True)
    elif args['force']:
        print(f'overwriting model directory `{visualization_dir}`')
    else:
        raise Exception(f'visualizatation directory `{visualization_dir}` already exists, use --force if you want to overwrite the folder')


    # Set random seed
    set_seed(args['seed'])  # Added here for reproductibility    
    
    w2i, i2w = args['dataset_class'].LABEL2INDEX, args['dataset_class'].INDEX2LABEL
    metrics_scores = []
    result_dfs = []
    # load model
    model, tokenizer, vocab_path, config_path = load_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    if args['fp16']:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16'])

    if args['device'] == "cuda":
        model = model.cuda()

    print("=========== TRAINING PHASE ===========")

    train_dataset_path = args['train_set_path']
    train_dataset = args['dataset_class'](train_dataset_path, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
    train_loader = args['dataloader_class'](dataset=train_dataset, max_seq_len=args['max_seq_len'], batch_size=args['train_batch_size'], num_workers=16, shuffle=False)  

    valid_dataset_path = args['valid_set_path']
    valid_dataset = args['dataset_class'](valid_dataset_path, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
    valid_loader = args['dataloader_class'](dataset=valid_dataset, max_seq_len=args['max_seq_len'], batch_size=args['valid_batch_size'], num_workers=16, shuffle=False)

    test_dataset_path = args['test_set_path']
    test_dataset = args['dataset_class'](test_dataset_path, tokenizer, lowercase=args["lower"], no_special_token=args['no_special_token'])
    test_loader = args['dataloader_class'](dataset=test_dataset, max_seq_len=args['max_seq_len'], batch_size=args['valid_batch_size'], num_workers=16, shuffle=False)

    
    # Train
    train(model, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], valid_criterion=args['valid_criterion'], i2w=i2w, n_epochs=args['n_epochs'], evaluate_every=1, early_stop=args['early_stop'], step_size=args['step_size'], gamma=args['gamma'], model_dir=model_dir, exp_id=0)

    # Save Meta
    if vocab_path:
        shutil.copyfile(vocab_path, f'{model_dir}/vocab.txt')
    if config_path:
        shutil.copyfile(config_path, f'{model_dir}/config.json')
        
    # Load best model
    model.load_state_dict(torch.load(model_dir + "/best_model_0.th"))

    # Evaluate
    print("=========== EVALUATION PHASE ===========")
    test_loss, test_metrics, test_hyp, test_label, test_seq = evaluate(model, data_loader=test_loader, forward_fn=args['forward_fn'], metrics_fn=args['metrics_fn'], i2w=i2w, is_test=True)

    print("test lost: ",test_loss)

    metrics_scores.append(test_metrics)
    result_dfs.append(pd.DataFrame({
        'seq':test_seq, 
        'hyp': test_hyp, 
        'label': test_label
    }))
    
    result_df = pd.concat(result_dfs)
    metric_df = pd.DataFrame.from_records(metrics_scores)
    
    print('== Prediction Result ==')
    print(result_df.head())
    print()
    
    print('== Model Performance ==')
    print(metric_df.describe())
    
    result_df.to_csv(model_dir + "/prediction_result.csv")
    metric_df.describe().to_csv(model_dir + "/evaluation_result.csv")