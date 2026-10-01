"""Pure mathematical transformations for FinPulse feature layer."""
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

def compute_log_amount(amount: float) -> float:
    """Safely compute log1p of non-negative amount."""
    return float(np.log1p(max(amount, 0.0)))

def compute_amount_to_balance_ratio(amount: float, origin_balance: float) -> float:
    """Safely compute amount to origin balance ratio."""
    return float(amount / (max(origin_balance, 0.0) + 1e-5))

def compute_amount_zscore(amount: float, mean: float, std: float) -> float:
    """
    Compute bounded behavioral Z-Score relative to customer historical mean.
    Clips between -5.0 and +10.0 to prevent outlier explosion.
    """
    std_safe = max(std, 1e-4)
    z = (amount - mean) / std_safe
    return float(np.clip(z, -5.0, 10.0))

def compute_cyclic_time(hour: int) -> Tuple[float, float]:
    """Compute continuous sin/cos cyclic representation of 24-hour clock."""
    angle = 2.0 * np.pi * (hour % 24) / 24.0
    return float(np.sin(angle)), float(np.cos(angle))

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS coordinates in kilometers."""
    if (lat1 == 0.0 and lon1 == 0.0) or (lat2 == 0.0 and lon2 == 0.0):
        return 0.0

    R = 6371.0 # Earth mean radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)

def compute_velocity_speed_kmh(distance_km: float, time_diff_seconds: float) -> float:
    """Calculate speed in km/h between two successive transactions to catch impossible travel."""
    if distance_km <= 0.0 or time_diff_seconds <= 0.0:
        return 0.0
    hours = max(time_diff_seconds / 3600.0, 1.0 / 3600.0)
    return float(distance_km / hours)

def compute_velocity_metrics(
    prior_events: List[Dict[str, Any]],
    current_timestamp: float
) -> Dict[str, Any]:
    """
    Pure calculation of rolling velocity counts and sums from prior events list.
    Prior events MUST be chronologically prior to current_timestamp (t < current_timestamp).
    Windows:
      - 1 minute (60s)
      - 5 minutes (300s)
      - 15 minutes (900s)
      - 1 hour (3600s)
    """
    count_1m = 0
    count_5m = 0
    count_15m = 0
    count_1h = 0
    sum_5m = 0.0
    sum_15m = 0.0
    sum_1h = 0.0

    t_1m = current_timestamp - 60.0
    t_5m = current_timestamp - 300.0
    t_15m = current_timestamp - 900.0
    t_1h = current_timestamp - 3600.0

    # Filter and accumulate across windows
    for ev in prior_events:
        ev_t = float(ev["timestamp"])
        if ev_t >= current_timestamp:
            continue # Ensure strict pre-decision ordering
        if ev_t >= t_1h:
            amt = float(ev.get("amount", 0.0))
            count_1h += 1
            sum_1h += amt

            if ev_t >= t_15m:
                count_15m += 1
                sum_15m += amt

            if ev_t >= t_5m:
                count_5m += 1
                sum_5m += amt

            if ev_t >= t_1m:
                count_1m += 1

    return {
        "tx_count_1m": count_1m,
        "tx_count_5m": count_5m,
        "tx_count_15m": count_15m,
        "tx_count_1h": count_1h,
        "amount_sum_5m": float(sum_5m),
        "amount_sum_15m": float(sum_15m),
        "amount_sum_1h": float(sum_1h)
    }

def compute_behavioral_metrics(
    prior_events: List[Dict[str, Any]],
    current_amount: float,
    current_category: str,
    current_hour: int,
    current_timestamp: float
) -> Dict[str, Any]:
    """
    Pure calculation of customer behavioral statistics over up to 30 days:
    user_avg_amount_30d, user_std_amount_30d, amount_zscore, user_category_frequency, user_hourly_tx_deviation.
    """
    t_30d = current_timestamp - (30.0 * 86400.0)
    amounts_30d: List[float] = []
    category_matches = 0
    hours_30d: List[int] = []

    for ev in prior_events:
        ev_t = float(ev["timestamp"])
        if ev_t >= current_timestamp:
            continue
        if ev_t >= t_30d:
            amt = float(ev.get("amount", 0.0))
            amounts_30d.append(amt)
            if str(ev.get("category", "")).lower() == str(current_category).lower():
                category_matches += 1
            hours_30d.append(int(ev.get("hour_of_day", ev.get("hour", 12))))

    n_prior = len(amounts_30d)
    if n_prior == 0:
        # First transaction: customer baseline equals current amount, 0 std, 0 zscore
        user_avg = float(current_amount)
        user_std = 0.0
        zscore = 0.0
        cat_freq = 1.0 # First occurrence is 100% of their history
        hourly_dev = 0.0
    elif n_prior == 1:
        user_avg = float(amounts_30d[0])
        user_std = 0.0
        # Single prior tx: measure difference normalized by baseline scale
        diff = current_amount - user_avg
        zscore = float(np.clip(diff / max(user_avg * 0.5, 1.0), -5.0, 10.0))
        cat_freq = float(category_matches / n_prior)
        h_diff = abs(current_hour - hours_30d[0])
        hourly_dev = float(min(h_diff, 24 - h_diff))
    else:
        user_avg = float(np.mean(amounts_30d))
        user_std = float(np.std(amounts_30d, ddof=1))
        zscore = compute_amount_zscore(current_amount, user_avg, user_std)
        cat_freq = float(category_matches / n_prior)
        mean_h = float(np.mean(hours_30d))
        h_diff = abs(current_hour - mean_h)
        hourly_dev = float(min(h_diff, 24.0 - h_diff))

    return {
        "user_avg_amount_30d": round(user_avg, 2),
        "user_std_amount_30d": round(user_std, 2),
        "amount_zscore": round(zscore, 4),
        "user_category_frequency": round(cat_freq, 4),
        "user_hourly_tx_deviation": round(hourly_dev, 2)
    }

def compute_spatial_and_device_metrics(
    prior_events: List[Dict[str, Any]],
    current_lat: float,
    current_lon: float,
    home_lat: float,
    home_lon: float,
    current_device_id: str,
    current_timestamp: float,
    device_user_events: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Pure calculation of spatial and device contextual metrics:
    distance_from_home_km, distance_from_prev_loc_km, speed_kmh_from_prev_tx,
    is_new_device, is_new_location, device_user_count_24h.
    """
    # 1. Distance from home
    dist_home = 0.0
    if (home_lat != 0.0 or home_lon != 0.0) and (current_lat != 0.0 or current_lon != 0.0):
        dist_home = haversine_distance_km(current_lat, current_lon, home_lat, home_lon)

    # 2. Previous location and speed
    dist_prev = 0.0
    speed = 0.0
    prior_locations = []
    prior_devices = set()

    # Filter and sort strictly prior events chronologically
    valid_prior = [ev for ev in prior_events if float(ev.get("timestamp", 0.0)) < current_timestamp]
    valid_prior.sort(key=lambda x: float(x.get("timestamp", 0.0)))

    for ev in valid_prior:
        ev_lat = float(ev.get("latitude", 0.0))
        ev_lon = float(ev.get("longitude", 0.0))
        if ev_lat != 0.0 or ev_lon != 0.0:
            prior_locations.append((ev_lat, ev_lon))
        dev = str(ev.get("device_id", "")).strip()
        if dev and dev != "unknown_device":
            prior_devices.add(dev)

    if valid_prior and (current_lat != 0.0 or current_lon != 0.0):
        last_event = valid_prior[-1]
        prev_lat = float(last_event.get("latitude", 0.0))
        prev_lon = float(last_event.get("longitude", 0.0))
        if prev_lat != 0.0 or prev_lon != 0.0:
            dist_prev = haversine_distance_km(current_lat, current_lon, prev_lat, prev_lon)
            time_diff = current_timestamp - float(last_event["timestamp"])
            speed = compute_velocity_speed_kmh(dist_prev, time_diff)

    # 3. New device detection
    cur_dev = str(current_device_id).strip()
    if not cur_dev or cur_dev == "unknown_device" or len(prior_devices) == 0:
        is_new_device = 0
    else:
        is_new_device = 1 if cur_dev not in prior_devices else 0

    # 4. New location detection (>50km from all prior locations)
    if (current_lat == 0.0 and current_lon == 0.0) or len(prior_locations) == 0:
        is_new_loc = 0
    else:
        min_dist_to_any_prior = min(
            haversine_distance_km(current_lat, current_lon, plat, plon)
            for plat, plon in prior_locations
        )
        is_new_loc = 1 if min_dist_to_any_prior > 50.0 else 0

    # 5. Device user count in last 24h
    device_user_count = 1
    if device_user_events is not None and cur_dev and cur_dev != "unknown_device":
        t_24h = current_timestamp - 86400.0
        unique_users = set()
        for dev_ev in device_user_events:
            dev_t = float(dev_ev.get("timestamp", 0.0))
            if t_24h <= dev_t <= current_timestamp:
                unique_users.add(str(dev_ev.get("customer_id", "")))
        if unique_users:
            device_user_count = max(len(unique_users), 1)

    return {
        "distance_from_home_km": round(dist_home, 2),
        "distance_from_prev_loc_km": round(dist_prev, 2),
        "speed_kmh_from_prev_tx": round(speed, 2),
        "is_new_device": int(is_new_device),
        "is_new_location": int(is_new_loc),
        "device_user_count_24h": int(device_user_count)
    }
