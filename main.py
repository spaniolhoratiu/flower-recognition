import test_accuracy_cnn_model


def main():
    print("Training model...")
    # train_and_build(200)
    while True:
        inputString = input('Input: ')
        if inputString == "predict image":
            print('prediction!')  # to be replaced with opening image, call prediction function, convert from int to
            # appropriate name
        else:
            if inputString == 'test accuracy':
                test_accuracy_cnn_model.test_accuracy()
            else:
                if inputString == 'exit':
                    inputString = input('Are you sure? [y/n]')
                    if inputString == 'y':
                        break
                else:
                    print(inputString)


if __name__ == "__main__":
    main()
