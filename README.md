To stabilize the pylint behaviour during the development process, it is recommended to set *python.defaultInterpreterPath* variable in .vscode/settings.json to python in your virtual environment.

Configs used for ablation are placed in the __***configs***__ folder.

```bash
python model_from_hf.py --config-path path/to/your/config/file
```

```bash
python model_from_hf.py --device "mps:0" \
                        --model-path path/to/model \
                        --dataset "tatsu-lab/alpaca" \
                        --sparsify \
                        --ablation-kind "begin" \
                        --sparse-implementation vanilla_k \
                        --lora-rank 2 \
                        --lora-alpha 4 \
                        --lora-dropout 0.05 \
                        --target-modules gate_proj up_proj \
                        down_proj k_proj v_proj o_proj q_proj \
                        --train-output-dir "results/res0/mt0-large-lora" \
                        --learning-rate 1e-3 \
                        --per-device-train-batch-size 2 \
                        --per-device-eval-batch-size 2 \
                        --num-train-epochs 2 \
                        --weight-decay 0.01 \
                        --hf-tasks hellaswag arc_challenge \
                        arc_easy boolq winogrande

```
___
Скорее всего, финальный отчет по проекту нужно будет оформить в отдельный документ. На данный момент все изменения вносятся в мидтермовский и отмечаются розовым хайлайтером.