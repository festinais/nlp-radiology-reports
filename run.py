mocked_args = """
    --model_name_or_path distilbert-base-cased
    --task_name mrpc
    --max_epochs 3
    --gpus 1""".split()

args = parse_args(mocked_args)
dm, model, trainer = main(args)
trainer.fit(model, dm)