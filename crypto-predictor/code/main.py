from fetch_data import fetch_bitcoin_data
from process_data import load_and_process
from model import train_model


def main():
    fetch_bitcoin_data()
    df = load_and_process()
    model = train_model(df)


if __name__ == "__main__":
    main()