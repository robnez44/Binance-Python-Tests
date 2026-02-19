from dataclasses import dataclass

@dataclass
class Metrics:
    start_index: int    # indice global (sobre toda la serie)
    end_index: int      # indice global (inclusivo)    
    lenght: int
    a: float            # pendiente
    b: float            # intercepto
    r2: float           
    pct_slope: float
    mean_price: float
    trend: str          # 'UP' / 'DOWN' / 'SIDE'

