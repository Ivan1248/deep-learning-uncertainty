
def write_all_text(path: str, text: str):
    with open(path, mode='w') as fs:
        fs.write(text)
        fs.flush()


def read_all_text(path: str) -> object:
    with open(path, mode='r') as fs:
        return fs.read()
    

def read_all_lines(path: str) -> object:
    with open(path, mode='r') as fs:
        return [line.strip() for line in fs]

if __name__ == "__main__":
    import os
    test_dir = os.path.join(os.path.dirname(__file__), "_test/")
    try:
        os.makedirs(test_dir)
    except:
        pass
    lines = ["bla", "bla", "blap"]
    path = os.path.join(test_dir, "test.txt")
    write_all_text(path, '\n'.join(lines))
    text = read_all_text(path)
    print(text)
    lines = read_all_lines(path)
    print(lines)