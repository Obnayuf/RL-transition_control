from evaluator import Evaluator

tmp = Evaluator()
tmp.load_saved_model(save_dir='/home/wrjs/Obnay/Lag/runs/BasicEnv-v1/CPO/seed-000-2023-04-15_15-55-25', model_name='model1799.pt')
tmp.evaluate()