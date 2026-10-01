"""CLI Entrypoint for Kafka Transaction Consumer Worker."""
import os
import sys
import argparse

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.streaming.config import load_streaming_config
from src.streaming.consumer import TransactionConsumer
from src.streaming.schema import TransactionEvent
from src.state.manager import RedisStateManager
from src.features.engine import FinPulseFeatureEngine
from src.models.inference import ProductionModelService

def transaction_handler(event: TransactionEvent, context=None, feature_result=None, inference_result=None):
    """
    R4 Real-Time Transaction Handler with Redis Historical Context, 32-Feature Extraction & ML Decisioning.
    Validates event arrival, displays state telemetry, 32-feature extraction vector summary, and ML decision.
    """
    flag = " [SIMULATED ANOMALY]" if event.is_fraud else ""
    ctx_info = ""
    if context and hasattr(context, "velocity"):
        vel_1h = context.velocity.get("tx_count_1h", 0)
        sum_1h = context.velocity.get("amount_sum_1h", 0.0)
        mean_30d = context.behavior.get("mean", 0.0) if hasattr(context, "behavior") else 0.0
        loc_str = "None"
        if getattr(context, "previous_location", None):
            loc_str = f"({context.previous_location['latitude']:.2f}, {context.previous_location['longitude']:.2f})"
        ctx_info = f" | Prior: [1h: {vel_1h}tx/${sum_1h:.1f}, 30d Mean: ${mean_30d:.1f}, PrevLoc: {loc_str}]"

    feat_info = ""
    if feature_result and hasattr(feature_result, "features"):
        features = feature_result.features
        feat_info = f" | R3 Features: [32-dim OK, log_amt={features[1]:.2f}, vel_1h={features[14]:.0f}, speed={features[25]:.1f}km/h]"

    ml_info = ""
    if inference_result and hasattr(inference_result, "decision"):
        ml_info = (
            f" | ML: [{inference_result.decision}] "
            f"(P_cal={inference_result.calibrated_probability:.4f}, "
            f"P_raw={inference_result.raw_probability:.4f}, "
            f"{inference_result.latency_ms:.1f}ms)"
        )

    print(
        f"[CONSUMED] Tx: {event.transaction_id} | "
        f"Cust: {event.customer_id} | "
        f"Amt: ${event.amount:8.2f} | "
        f"Cat: {event.category:<14} | "
        f"Payment: {event.payment_type:<11} | "
        f"Auth: {'OK' if event.auth_verified else 'FAIL'}{flag}{ctx_info}{feat_info}{ml_info}"
    )

def main():
    parser = argparse.ArgumentParser(description="FinPulse Real-Time Kafka Transaction Consumer")
    parser.add_argument("--max-messages", type=int, default=None, help="Stop after consuming N messages")
    parser.add_argument("--group-id", type=str, default=None, help="Override Kafka consumer group ID")
    parser.add_argument("--bootstrap-servers", type=str, default=None, help="Override Kafka bootstrap servers")
    parser.add_argument("--auto-offset-reset", type=str, default=None, choices=["earliest", "latest"], help="Auto offset reset")
    parser.add_argument("--no-redis", action="store_true", help="Disable Redis historical state manager")
    parser.add_argument("--no-features", action="store_true", help="Disable R3 32-feature extraction engine")
    parser.add_argument("--no-inference", action="store_true", help="Disable R4 real-time CatBoost inference & decisioning")
    args = parser.parse_args()

    config = load_streaming_config()
    if args.bootstrap_servers:
        config.bootstrap_servers = args.bootstrap_servers
    if args.group_id:
        config.consumer.group_id = args.group_id
    if args.auto_offset_reset:
        config.consumer.auto_offset_reset = args.auto_offset_reset

    state_manager = None if args.no_redis else RedisStateManager()
    feature_engine = None if args.no_features else FinPulseFeatureEngine(state_manager=state_manager)
    model_service = None if args.no_inference else ProductionModelService()

    print("=" * 80)
    print("  FINPULSE REAL-TIME KAFKA TRANSACTION CONSUMER (R4 ML INFERENCE ACTIVE)")
    print("=" * 80)
    print(f"Bootstrap Servers: {config.bootstrap_servers}")
    print(f"Inbound Topic:     {config.topics.inbound_transactions}")
    print(f"Consumer Group:    {config.consumer.group_id}")
    print(f"Offset Reset:      {config.consumer.auto_offset_reset}")
    print(f"Manual Commit:     {not config.consumer.enable_auto_commit}")
    print(f"Redis State:       {'Disabled' if args.no_redis else ('Connected OK' if state_manager and state_manager.health_check() else 'Degraded/Mock')}")
    print(f"Feature Engine:    {'Disabled' if args.no_features else 'Active (32-Feature Vector)'}")
    print(f"Model Inference:   {'Disabled' if args.no_inference else f'Active ({model_service.model_version}, Platt Calibrated)'}")
    print(f"Max Messages:      {args.max_messages or 'Infinite (Ctrl+C to stop)'}")
    print("-" * 80)

    consumer = TransactionConsumer(
        config=config,
        handler=transaction_handler,
        state_manager=state_manager,
        feature_engine=feature_engine,
        model_service=model_service
    )

    try:
        consumer.run_loop(max_messages=args.max_messages)
    finally:
        consumer.close()
        if state_manager:
            state_manager.close()

if __name__ == "__main__":
    main()
