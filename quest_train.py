import logging
import train

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Calling train.main()")
    train.main()
    logging.info("Training finished")