import os

from dotenv import load_dotenv

from db import init_db

load_dotenv()



if __name__ == "__main__":

    print("Initializing the database!")
    init_db()
