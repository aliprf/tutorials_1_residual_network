from train import Train


if __name__ == '__main__':
    trainer = Train(input_shape=[28, 28, 1], output_shape=10)
    trainer.train_model()