from datetime import datetime, timezone
import pytz

timestamp_ms = 1770825600000

timestamp_s = timestamp_ms / 1000

date_utc = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
print("UTC:", date_utc)

timez = pytz.timezone("America/El_Salvador")
local_date = date_utc.astimezone(timez)
print("Hora El Salvador:", local_date)
