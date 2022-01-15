python main.py \
    --n_epochs=25 \
    --train_batch_size=8 \
    --model_checkpoint=indobenchmark/indobert-lite-large-p1 \
    --step_size=1 \
    --gamma=0.9 \
    --device=cuda \
    --experiment_name=indobert-lite-large-p1-uncased_b32_step1_gamma0.9_lr1e-5_early3_layer24_lowerTrue \
    --lr=1e-5 \
    --early_stop=12 \
    --dataset=qa-factoid-itb \
    --lower \
    --num_layers=24 \
    --max_norm=10 \
    --seed=42 \
    --force