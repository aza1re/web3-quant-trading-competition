API_KEY="p0SVYNgXn1NDDHVPsVjeUF4OZiLJOdEJZ7R4Q4cMF34HBglbQ3ed6hLOSnuv6t4a"
API_SECRET="w2sLJWgFWMQLRaBsZT8q8pdivtzEnUzCuol38ElvMthqbRQjkJx85uHduoDXTCTA"
YOUR_HORUS_KEY="a0ff981638cc60f41d91bcd588b782088d28d04a614a8ad633cee70f660b967a"
#pip install -r requirements.txt
#python main.py --deploy "$API_KEY" "$API_SECRET"
python qc_converted/main.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1d --limit 30 --apikey "$YOUR_HORUS_KEY" --capital 50000 --verbose --risk-mult 10.0 --tighter-stops