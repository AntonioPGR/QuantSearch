from dataclasses import dataclass


@dataclass
class Config:
    # DIRECTORIES
    DATA_DIR = "_DATA/featured/"

    # STOCKS
    STOCKS = ['VALE3', 'PETR4', 'AXIA3', 'PETR3', 'ITSA4', 'SBSP3', 'WEGE3', 'EMBJ3', 'CPLE3', 'GGBR4']
    PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


    # TRAIN TEST SIZES
    TRAIN_DAYS: int = 3
    TEST_DAYS: int = 1
    STEP_DAYS: int = 1


    # TRAIN CONFIGS
    EPISODES: int = 8
    UPDATE_TIMESTAMP: int = 256
    # initial_capital: float = 10_000.0