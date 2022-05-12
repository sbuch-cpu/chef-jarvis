import pickle

def main():
    file = read_from_pickle("./training_data/27/gen_boxes/6_3.pickle")
    print(file)
    return


def read_from_pickle(path):
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects

if __name__ == "__main__":
    main()
