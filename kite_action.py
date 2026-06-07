
from kiteconnect import KiteTicker

def start_ws_if_needed():
    global WS_RUNNING, WS_THREAD
    log1("ws will start here")
    with ws_lock:
        if WS_RUNNING:
            return
        WS_RUNNING = True
        access_token = read_access_token()

        def run_ws():
            try:
                kite_local = KiteConnect(api_key=API_KEY)
                kite_local.set_access_token(access_token)

                start_ws(API_KEY, access_token, kite_local)
                log1("ws will started here ------------")
            except Exception as e:
                log1(f"WS error: {e}")
            finally:
                WS_RUNNING = False

        WS_THREAD = threading.Thread(target=run_ws)
        WS_THREAD.daemon = True
        WS_THREAD.start()

        log1("🚀 WebSocket started")


def stop_ws_if_idle():
    global WS_RUNNING, KWS

    with ws_lock, threads_lock:
        active = [
            t for t in TRAILING_THREADS.values()
            if t.is_alive()
        ]

        if len(active) == 0:
            if KWS:
                try:
                    KWS.close()   # ✅ ACTUAL STOP
                    log1("🛑 WebSocket closed")
                except Exception as e:
                    log1(f"WS close error: {e}")

            WS_RUNNING = False
            KWS = None
            log1("🛑 WebSocket stopping (no active threads)")

def start_ws(api_key, access_token, kite):

    global KWS

    kws = KiteTicker(api_key, access_token)
    KWS = kws   # ✅ STORE INSTANCE

    def on_connect(ws, response):

        print("✅ WebSocket Connected")

        tokens = list(set(
            int(v)
            for v in INSTRUMENT_MAP.values()
            if v
        ))

        log1(f"📡 Subscribing {len(tokens)} tokens")

        # 🔥 Chunking (VERY IMPORTANT)
        chunk_size = 3000

        for i in range(0, len(tokens), chunk_size):

            chunk = tokens[i:i + chunk_size]

            ws.subscribe(chunk)

            ws.set_mode(ws.MODE_LTP, chunk)

        log1("✅ WebSocket subscription completed")

    def on_ticks(ws, ticks):
        # log1("in on_ticks loop")
        for tick in ticks:
            token = tick["instrument_token"]
            ltp = tick["last_price"]
            with sl_lock:
                items = list(LIVE_SL.items())  # copy for safe iteration

            for task_id, data in items:
                if INSTRUMENT_MAP.get(data["symbol"]) != token:
                    continue

                sl = data["stoploss"]
                position = data["position"]
                qty = data["qty"]
                symbol = data["symbol"]
                # log1(f"{data} tick data ------------- {ltp}")

                try:
                    # 🚨 EXIT LOGIC
                    if position == 1 and ltp <= sl:
                        log1(f"[{task_id}] 🔥 SL HIT LONG {symbol} at {ltp}")

                        exit_market(kite, symbol, qty, "SELL")

                        with sl_lock:
                            LIVE_SL.pop(task_id, None)

                        delete_live_sl(task_id)
                        stop_task(task_id, "SL Hit") 

                    elif position == -1 and ltp >= sl:
                        log1(f"[{task_id}] 🔥 SL HIT SHORT {symbol} at {ltp}")

                        exit_market(kite, symbol, qty, "BUY")

                        with sl_lock:
                            LIVE_SL.pop(task_id, None)

                        delete_live_sl(task_id)
                        stop_task(task_id, "SL Hit") 

                except Exception as e:
                    log1(f"[{task_id}] Exit error: {e}")
    def on_close(ws, code, reason):
        global WS_RUNNING, KWS
        log1(f"🔌 WS Closed: {reason}")
        WS_RUNNING = False
        KWS = None

    kws.on_close = on_close
    kws.on_connect = on_connect
    def combined_ticks(ws, ticks):

        # 🔥 LIVE LTP
        on_ticks_ltp(ws, ticks)

        # 🔥 SL LOGIC
        on_ticks(ws, ticks)

    kws.on_ticks = combined_ticks

    kws.connect(threaded=True)
def exit_market(kite, symbol, qty, side):
    log1(f"in exit condition -- {symbol} -- {side} -- {qty}")
    try:
        txn_type = kite.TRANSACTION_TYPE_SELL if side == "SELL" else kite.TRANSACTION_TYPE_BUY

        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_MCX,
            tradingsymbol=symbol,
            transaction_type=txn_type,
            quantity=qty,
            product=kite.PRODUCT_NRML,
            order_type=kite.ORDER_TYPE_MARKET,
            market_protection=2,
            validity=kite.VALIDITY_DAY
        )

        log1(f"✅ EXITED {symbol}, Order ID: {order_id}")

    except Exception as e:
        log1(f"Exit error: {e}")


order_id = kite_obj.place_order(
            variety=kite_obj.VARIETY_REGULAR,
            exchange=kite_obj.EXCHANGE_MCX,
            tradingsymbol=tradingsymbol,
            transaction_type=kite_obj.TRANSACTION_TYPE_SELL,
            quantity=quantity,
            product=kite_obj.PRODUCT_NRML,
            order_type=kite_obj.ORDER_TYPE_SLM,   # 👈 IMPORTANT
            trigger_price=stoploss_val,
            validity=kite_obj.VALIDITY_DAY
        )

order_id = kite_obj.place_order(
            variety=kite_obj.VARIETY_REGULAR,
            exchange=kite_obj.EXCHANGE_MCX,
            tradingsymbol=tradingsymbol,
            transaction_type=kite_obj.TRANSACTION_TYPE_BUY,
            quantity=quantity,
            product=kite_obj.PRODUCT_NRML,
            order_type=kite_obj.ORDER_TYPE_SLM,
            trigger_price=stoploss_val,
            market_protection=2,
            validity=kite_obj.VALIDITY_DAY
        )


def modify_sl_order(sl_orderid, stoploss_val, transaction, kite_instance=None):
    global kite

    kite_obj = kite_instance if kite_instance is not None else kite

    if not kite_obj:
        raise Exception("❌ Kite instance not available")

    return kite_obj.modify_order(
        variety=kite_obj.VARIETY_REGULAR,
        order_id=sl_orderid,
        trigger_price=stoploss_val,
        market_protection=2
    )

def cancel_order(orderid, kite_instance=None):
    global kite

    kite_obj = kite_instance if kite_instance is not None else kite

    if not kite_obj:
        raise Exception("❌ Kite instance not available")

    try:
        return kite_obj.cancel_order(
            variety=kite_obj.VARIETY_REGULAR,
            order_id=orderid
        )
    except Exception as e:
        log(f"❌ Stoploss cancel error: {e}")


def kite_app_buy_sell(exchange, tradingsymbol, buy_sell, quantity, kite_instance=None):
    global kite

    kite_obj = kite_instance if kite_instance is not None else kite

    if not kite_obj:
        raise Exception("❌ Kite instance not available")

    order_id = kite_obj.place_order(
        variety=kite_obj.VARIETY_REGULAR,
        exchange=exchange,
        tradingsymbol=tradingsymbol,
        transaction_type=buy_sell,
        quantity=quantity,
        order_type=kite_obj.ORDER_TYPE_MARKET,
        product=kite_obj.PRODUCT_NRML,
        validity=kite_obj.VALIDITY_DAY
    )

    log(f"✅ {buy_sell} ORDER | {tradingsymbol} | Qty={quantity} | OrderID={order_id}")

    return order_id