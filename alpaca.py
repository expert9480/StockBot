# from alpaca.trading.client import TradingClient
# from alpaca.trading.requests import GetAssetsRequest
# from alpaca.trading.enums import AssetClass
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os

alpacaBaseURL = "https://paper-api.alpaca.markets"

load_dotenv()

#new api
# # paper=True enables paper trading
# trading_client = TradingClient(os.getenv('alpacaKey'), os.getenv('alpacaSecret'), paper=True)
# # account = trading_client.get_account()
# search_params = GetAssetsRequest(asset_class=AssetClass.CRYPTO)
# assets = trading_client.get_all_assets(search_params)
# print(assets)

alpaca = tradeapi.REST(os.getenv('alpacaKey'), os.getenv('alpacaSecret'), alpacaBaseURL, 'v2')
orders = alpaca.list_orders(status="open")
print(orders)

def checkIfMarketOpen():
    clock = alpaca.get_clock()
    return clock.is_open # True or False

print(alpaca.get_asset("CSCO"))
print(alpaca.get_bars("CSCO", tradeapi.TimeFrame.Day, "2026-01-07", "2026-01-11").df) #day behind?
print(alpaca.get_latest_trade("CSCO"))

#market price
# order1 = alpaca.submit_order(
#     symbol="NVDA",
#     qty=1,
#     side="buy",
#     type="market",
#     time_in_force="gtc"
# )

#limit price
# order2 = alpaca.submit_order(
#     symbol="NVDA",
#     qty=1,
#     side="buy",
#     type="limit",
#     limit_price=170.00,  # set your desired price
#     time_in_force="gtc"
# )




