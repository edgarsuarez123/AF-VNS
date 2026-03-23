import wfdb
import os
os.makedirs("data/raw/stroke avns/bidmc", exist_ok=True)
wfdb.dl_database("bidmc", "data/raw/stroke avns/bidmc")
print("Done")
