"""Realistic evaluation queries. None of these sentences appear in data/intents.json."""

# (query, expected intent), asked in a fresh conversation
SINGLE = [
    # order status / tracking
    ("can you tell me where my parcel is", "order_status"), ("i want to know the delivery status of my order", "order_status"),
    ("track my package", "order_status"), ("status of order 10001", "order_status"), ("has my order been dispatched", "order_status"),
    ("when will i receive my order", "order_status"), ("tracking number please", "order_status"), ("order tracking", "order_status"),
    # cancel
    ("i want to cancel", "cancel_order"), ("please cancel order 10002", "cancel_order"), ("how can i cancel my order", "cancel_order"),
    ("cancel", "cancel_order"), ("i don't want this order anymore", "cancel_order"),
    # shipping
    ("how much is delivery", "shipping_info"), ("do you deliver internationally", "shipping_info"), ("what is the shipping fee", "shipping_info"),
    ("how many days will delivery take", "shipping_info"), ("shipping", "shipping_info"), ("delivery charges", "shipping_info"),
    ("do you offer same day delivery", "shipping_info"), ("how long does shipping take", "shipping_info"),
    # refund policy
    ("what is the refund policy", "refund_policy"), ("can i get my money back", "refund_policy"), ("refund", "refund_policy"),
    ("do you offer refunds", "refund_policy"), ("i want a refund", "refund_policy"), ("under what conditions can i get a refund", "refund_policy"),
    # refund status
    ("i still haven't got my refund", "refund_status"), ("when will my refund be processed", "refund_status"), ("refund status", "refund_status"),
    ("my refund has not come yet", "refund_status"), ("how many days for the refund to reflect", "refund_status"),
    # returns
    ("how do i return a product", "return_process"), ("i want to return this item", "return_process"), ("return", "return_process"),
    ("what is your return policy", "return_process"), ("can i return something i bought", "return_process"), ("how to exchange a product", "return_process"),
    ("i would like to send my order back", "return_process"),
    # payments
    ("what payment options are available", "payment_methods"), ("can i pay using upi", "payment_methods"), ("do you take credit cards", "payment_methods"),
    ("is cash on delivery available", "payment_methods"), ("payment methods", "payment_methods"), ("can i pay in installments", "payment_methods"),
    ("payment", "payment_methods"),
    # payment issue
    ("my payment failed", "payment_issue"), ("i was charged twice for the same order", "payment_issue"), ("money deducted but order not confirmed", "payment_issue"),
    ("payment went through but no order id", "payment_issue"), ("transaction failed but amount debited", "payment_issue"),
    ("i did not get my order confirmation", "payment_issue"), ("payment stuck", "payment_issue"),
    # address
    ("i want to change my delivery address", "change_address"), ("can i update the address on my order", "change_address"), ("wrong address", "change_address"),
    ("send it to a different address", "change_address"),
    # password / account
    ("i forgot my password", "reset_password"), ("how do i reset my password", "reset_password"), ("can't login to my account", "reset_password"),
    ("password", "reset_password"), ("how do i sign up", "create_account"), ("create account", "create_account"), ("i want to register", "create_account"),
    ("change my email id", "account_update"), ("delete my account", "account_update"), ("update my phone number", "account_update"),
    # promo
    ("do you have any coupon codes", "promo_codes"), ("is there any discount", "promo_codes"), ("coupon", "promo_codes"), ("how do i use a promo code", "promo_codes"),
    ("any offers going on", "promo_codes"),
    # warranty / defective / wrong item
    ("is there any warranty on this", "warranty"), ("my phone stopped working", "warranty"), ("the product is damaged", "warranty"),
    ("warranty", "warranty"), ("received a damaged item", "warranty"),
    ("i got the wrong product", "wrong_or_missing_item"), ("item missing in my package", "wrong_or_missing_item"), ("received a different size", "wrong_or_missing_item"),
    # delayed
    ("my order is delayed", "delayed_delivery"), ("why is my delivery so late", "delayed_delivery"), ("it's been 10 days and i haven't received my order", "delayed_delivery"),
    ("order not arrived yet", "delayed_delivery"),
    # invoice / stock / modify / gift
    ("i need the invoice", "invoice"), ("how do i download my bill", "invoice"), ("is this item available", "stock_availability"),
    ("out of stock", "stock_availability"), ("when will this be restocked", "stock_availability"), ("can i change my order", "modify_order"),
    ("i want to add another item to my order", "modify_order"), ("gift card", "gift_cards"), ("how do i redeem a gift voucher", "gift_cards"),
    # support / hours / human / bot
    ("how do i contact you", "contact_support"), ("what is your phone number", "contact_support"), ("customer care number", "contact_support"),
    ("email address for support", "contact_support"), ("contact", "contact_support"),
    ("what are your working hours", "business_hours"), ("are you open today", "business_hours"), ("timings", "business_hours"),
    ("i want to talk to a person", "human_agent"), ("connect me to an agent", "human_agent"), ("agent", "human_agent"), ("human", "human_agent"),
    ("what can you help me with", "bot_identity"), ("are you a real person", "bot_identity"), ("help", "bot_identity"),
    # small talk
    ("hi there", "greeting"), ("hello", "greeting"), ("good evening", "greeting"), ("hey", "greeting"),
    ("thanks", "thanks"), ("thank you", "thanks"), ("ok thanks", "thanks"), ("that was helpful thanks", "thanks"),
    ("bye", "goodbye"), ("goodbye", "goodbye"), ("see you", "goodbye"),
]

# multi-turn scenarios: list of (user message, expected intent) played in ONE conversation
SCENARIOS = {
    "refund then follow-ups": [("what is the refund policy", "refund_policy"), ("how long will it take", "followup_duration"), ("is it free", "followup_cost"), ("tell me more", "followup_more")],
    "shipping follow-ups": [("do you ship internationally", "shipping_info"), ("how much does it cost", "followup_cost"), ("and how long", "followup_duration")],
    "return then refund question": [("how do i return an item", "return_process"), ("is it free", "followup_cost"), ("when will i get my refund", "refund_status")],
    "topic switch mid-chat": [("what is your refund policy", "refund_policy"), ("do you have any coupons", "promo_codes"), ("how long does delivery take", "shipping_info"),
                               ("how do i reset my password", "reset_password")],
    "explicit topic beats follow-up cue": [("what is your refund policy", "refund_policy"), ("how long does delivery take", "shipping_info"),
                                           ("how many days will the refund take", "refund_status"), ("how long does a return take", "return_process")],
    "explicit topic in 'how long' (after other topic)": [("do you accept paypal", "payment_methods"), ("how long will my refund take", "refund_status")],
    "payment trouble": [("my payment failed", "payment_issue"), ("how long will it take", "followup_duration"), ("will i get my money back", "refund_policy", "automatically")],
    "order then address": [("where is my order", "order_status"), ("10002", "order_status"), ("can i change the address", "change_address"), ("how long will it take", "followup_duration")],
    "order lookup then unrelated question": [("track my order", "order_status"), ("what payment methods do you accept", "payment_methods"), ("10001", "order_status")],
    "pending order id, user asks something else": [("cancel my order", "cancel_order"), ("what is your return policy", "return_process"), ("thanks", "thanks")],
    "cancel flow with question in the middle": [("cancel order 10002", "cancel_order"), ("what is the refund policy", "refund_policy"), ("yes", "cancel_order")],
    "yes/no without context": [("yes", "acknowledge"), ("no", "acknowledge")],
    "small talk does not reset topic": [("what is your refund policy", "refund_policy"), ("thanks", "thanks"), ("how long will it take", "followup_duration")],
    "name then question": [("my name is Priya", "provide_name"), ("what are your business hours", "business_hours"), ("thank you", "thanks")],
    "warranty to wrong item": [("my headphones stopped working", "warranty"), ("is it free", "followup_cost"), ("i got the wrong item", "wrong_or_missing_item")],
}

# (previous topic message, follow-up, expected substring in the follow-up reply)
FOLLOWUP_TEXT = [
    ("what is your refund policy", "how long will it take", "5-7"),
    ("do you ship internationally", "how long will it take", "3-5"),
    ("do you ship internationally", "is it free", "free over"),
    ("how do i cancel my order", "is it free", "no cancellation fee"),
    ("my payment failed", "how long will it take", "2-4 hours"),
    ("how do i return an item", "how long will it take", "Pickup"),
    ("how many days will the refund take", None, "5-7"),
    ("how long does a return take", None, "Pickup"),
]


# ---- Held-out set: written BEFORE tuning and never used to tune. Measures real generalisation. ----
HELDOUT = [
    ("where is my shipment", "order_status"), ("is my order on the way", "order_status"), ("i placed an order last week, where is it", "order_status"),
    ("check delivery status", "order_status"), ("my tracking link is not working", "order_status"),
    ("i need to cancel an order i just placed", "cancel_order"), ("can you cancel it for me", "cancel_order"), ("cancel my purchase please", "cancel_order"),
    ("what are the shipping charges to mumbai", "shipping_info"), ("how fast is express delivery", "shipping_info"), ("do you deliver on sundays", "shipping_info"),
    ("is delivery free", "shipping_info"),
    ("what's your refund process", "refund_policy"), ("am i eligible for a refund", "refund_policy"), ("refund conditions", "refund_policy"),
    ("my refund is pending", "refund_status"), ("where's my refund", "refund_status"), ("refund not received", "refund_status"),
    ("how can i return a defective item", "return_process"), ("return an order", "return_process"), ("i need to exchange for a bigger size", "return_process"),
    ("which payment modes do you support", "payment_methods"), ("can i use netbanking", "payment_methods"), ("do you have emi", "payment_methods"),
    ("my card got declined", "payment_issue"), ("amount deducted twice", "payment_issue"), ("paid but order not placed", "payment_issue"),
    ("i didn't receive a confirmation email", "payment_issue"),
    ("i put the wrong pincode", "change_address"), ("edit shipping address", "change_address"),
    ("reset password link not coming", "reset_password"), ("unable to log in", "reset_password"), ("how to open an account", "create_account"),
    ("i want to update my profile", "account_update"), ("stop sending me emails", "account_update"),
    ("any promo codes", "promo_codes"), ("is there a sale going on", "promo_codes"), ("my coupon is not applying", "promo_codes"),
    ("what does the warranty cover", "warranty"), ("my laptop is not turning on", "warranty"), ("item is faulty", "warranty"),
    ("i got someone else's order", "wrong_or_missing_item"), ("one item is missing from my box", "wrong_or_missing_item"), ("this is not the product i ordered", "wrong_or_missing_item"),
    ("my package is stuck in transit", "delayed_delivery"), ("delivery is taking forever", "delayed_delivery"), ("it was supposed to arrive yesterday", "delayed_delivery"),
    ("send me the gst invoice", "invoice"), ("where can i find my receipt", "invoice"), ("is the red one in stock", "stock_availability"),
    ("notify me when it is back in stock", "stock_availability"), ("i want to change the colour of my order", "modify_order"),
    ("how do i use my gift card", "gift_cards"), ("do you have gift vouchers", "gift_cards"),
    ("what is your email id", "contact_support"), ("how can i reach customer care", "contact_support"),
    ("when does support open", "business_hours"), ("are you open on weekends", "business_hours"), ("what time do you close", "business_hours"),
    ("get me a real person", "human_agent"), ("i want to speak to your manager", "human_agent"), ("live chat with agent", "human_agent"),
    ("are you an ai", "bot_identity"), ("what do you do", "bot_identity"),
    ("good morning", "greeting"), ("hi, i need some help", "bot_identity"), ("thanks a lot", "thanks"), ("ok bye", "goodbye"),
    # typos
    ("wheres my ordr", "order_status"), ("refud policy", "refund_policy"), ("cancle my order", "cancel_order"), ("shiping charges", "shipping_info"),
    ("paymnet failed", "payment_issue"), ("i forgott my pasword", "reset_password"),
]

HELDOUT_SCENARIOS = {
    "return flow": [("i want to return a product", "return_process"), ("how long does it take", "followup_duration"), ("is there any charge", "followup_cost"),
                    ("when will i get the money back", "refund_status")],
    "shipping explicit follow-up": [("what is your refund policy", "refund_policy"), ("what are the shipping charges", "shipping_info"), ("how long will delivery take", "shipping_info")],
    "coupon then checkout": [("do you have a coupon", "promo_codes"), ("how do i apply it", "promo_codes"), ("what payment methods can i use", "payment_methods")],
    "order then cancel": [("track order 10002", "order_status"), ("cancel it", "cancel_order"), ("yes", "cancel_order"), ("what about the refund", "refund_policy")],
    "interrupted order lookup": [("where is my order", "order_status"), ("actually how do i return something", "return_process"), ("what is the refund policy", "refund_policy")],
    "support chain": [("my product is faulty", "warranty"), ("how long will it take", "followup_duration"), ("i want to talk to someone", "human_agent"), ("thanks", "thanks")],
}


# ---- Fresh set C: written after the fixes, never used for tuning ----
FRESH = [
    ("where's my stuff", "order_status"), ("what happened to my order", "order_status"), ("track order number 10003", "order_status"),
    ("i ordered a watch, when will it come", "order_status"), ("can i see where my package is right now", "order_status"),
    ("please stop my order", "cancel_order"), ("i accidentally ordered the wrong thing, cancel it", "cancel_order"), ("order cancellation", "cancel_order"),
    ("do you charge for delivery", "shipping_info"), ("how many days does standard shipping take", "shipping_info"), ("can you ship to dubai", "shipping_info"),
    ("what are the delivery options", "shipping_info"),
    ("i would like to know about your refund rules", "refund_policy"), ("will you refund me if i don't like it", "refund_policy"), ("full refund possible", "refund_policy"),
    ("i returned my order two weeks ago and still nothing", "refund_status"), ("has my refund been issued", "refund_status"), ("refund where", "refund_status"),
    ("how do i send an item back", "return_process"), ("can i swap this for another one", "return_process"), ("generate a return label", "return_process"),
    ("pay by card", "payment_methods"), ("do you accept apple pay", "payment_methods"), ("is there pay on delivery", "payment_methods"),
    ("the payment didn't go through", "payment_issue"), ("my card was charged but i got no confirmation", "payment_issue"), ("i was billed two times", "payment_issue"),
    ("can you change where it's being sent", "change_address"), ("my address is incorrect", "change_address"),
    ("i can't remember my password", "reset_password"), ("sign in is not working", "reset_password"), ("how do i make an account", "create_account"),
    ("change my registered mobile number", "account_update"), ("close my account", "account_update"),
    ("got any discount codes", "promo_codes"), ("first order discount", "promo_codes"), ("how to apply coupon at checkout", "promo_codes"),
    ("is my product covered under warranty", "warranty"), ("headphones are not working", "warranty"), ("the screen is cracked", "warranty"),
    ("received an incorrect item", "wrong_or_missing_item"), ("you forgot to include the charger", "wrong_or_missing_item"), ("package arrived with missing items", "wrong_or_missing_item"),
    ("my parcel hasn't arrived and it's overdue", "delayed_delivery"), ("delivery date passed", "delayed_delivery"), ("why is it taking so long to arrive", "delayed_delivery"),
    ("can i get a copy of my bill", "invoice"), ("i need a tax invoice", "invoice"), ("when is this back in stock", "stock_availability"),
    ("is it available in medium", "stock_availability"), ("i need to change the size on my order", "modify_order"), ("can i edit my order", "modify_order"),
    ("where do i enter my gift card", "gift_cards"), ("can i buy a gift voucher", "gift_cards"),
    ("what's your helpline", "contact_support"), ("how do i get in touch with you", "contact_support"), ("what days are you open", "business_hours"),
    ("support hours", "business_hours"), ("let me talk to a human please", "human_agent"), ("is there someone i can speak to", "human_agent"),
    ("are you a bot", "bot_identity"), ("what are you", "bot_identity"), ("hello there", "greeting"), ("hi, anyone there", "greeting"),
    ("thanks for the help", "thanks"), ("thank you so much", "thanks"), ("goodbye then", "goodbye"), ("that's all, bye", "goodbye"),
    # misspellings and shorthand
    ("wher is my oder", "order_status"), ("i want a refnd", "refund_policy"), ("cancel my ordr", "cancel_order"), ("pasword reset", "reset_password"),
    ("shipng time", "shipping_info"), ("retrun policy", "return_process"), ("paymnt failed", "payment_issue"), ("my pakage is lost", "delayed_delivery"),
    ("pls help me track order", "order_status"), ("hw do i return", "return_process"),
]

FRESH_SCENARIOS = {
    "return, then ask about the refund": [("i want to return my shoes", "return_process"), ("how long does it take", "followup_duration"),
                                          ("and how many days for the refund", "refund_status"), ("thanks", "thanks")],
    "shipping topic stays on shipping": [("how much is shipping", "shipping_info"), ("how long will it take", "followup_duration", "3-5"),
                                         ("is it free", "followup_cost", "free over")],
    "jump between topics": [("how do i reset my password", "reset_password"), ("what's your refund policy", "refund_policy"), ("how long does delivery take", "shipping_info"),
                            ("how long does a refund take", "refund_status")],
    "order lookup, then address, then duration": [("where is my order 10001", "order_status"), ("i want to change the address", "change_address"),
                                                  ("how long will it take", "followup_duration", "24 hours")],
    "cancel with an interruption": [("cancel order 10002", "cancel_order"), ("do you accept upi", "payment_methods", "still want me to cancel"), ("yes", "cancel_order", "Done")],
    "cancel then decline": [("i want to cancel order 10002", "cancel_order"), ("no", "cancel_order", "haven't changed")],
    "ordering questions with a name": [("my name is Ravi", "provide_name"), ("track my order", "order_status"), ("10004", "order_status", "out for delivery")],
    "ok inside a question must not confirm a cancel": [("cancel order 10002", "cancel_order"), ("ok what is your refund policy", "refund_policy", "still want me to cancel"),
                                                       ("yes", "cancel_order", "Done")],
    "payment failed chain": [("my payment failed", "payment_issue"), ("will i get my money back", "refund_policy", "automatically"), ("how long will it take", "followup_duration")],
    "gibberish then recovery": [("asdfgh", None), ("what is your refund policy", "refund_policy")],
}


# ---- Set D: written last, scored once, never tuned on ----
UNSEEN = [
    ("hey, can you check on my delivery", "order_status"), ("status of my purchase please", "order_status"), ("when is my item getting here", "order_status"),
    ("i haven't gotten any tracking info", "order_status"), ("is order 10004 out for delivery", "order_status"), ("let me know where my order is", "order_status"),
    ("i changed my mind, don't want it", "cancel_order"), ("how do i cancel something i ordered yesterday", "cancel_order"), ("cancel the headphones order", "cancel_order"),
    ("is there a delivery fee", "shipping_info"), ("how long until it ships", "shipping_info"), ("do you do express shipping", "shipping_info"),
    ("do you send parcels abroad", "shipping_info"),
    ("what's the policy on refunds", "refund_policy"), ("i want my money returned", "refund_policy"), ("are purchases refundable", "refund_policy"),
    ("it's been a week and my refund isn't here", "refund_status"), ("when do i get my refund", "refund_status"), ("any update on my refund", "refund_status"),
    ("i want to send this back", "return_process"), ("what's involved in returning an item", "return_process"), ("exchange for a different colour", "return_process"),
    ("can i pay with a credit card", "payment_methods"), ("what forms of payment do you accept", "payment_methods"), ("is gpay accepted", "payment_methods"),
    ("my transaction failed but money was cut", "payment_issue"), ("i was double charged", "payment_issue"), ("no order confirmation after paying", "payment_issue"),
    ("the card payment keeps failing", "payment_issue"),
    ("i typed the wrong street address", "change_address"), ("update delivery location", "change_address"),
    ("forgot my login details", "reset_password"), ("can't get into my account", "reset_password"), ("i want to open a new account", "create_account"),
    ("how do i change my email", "account_update"), ("remove my account", "account_update"), ("unsubscribe me from the newsletter", "account_update"),
    ("do you have a promo code for new users", "promo_codes"), ("any ongoing sales", "promo_codes"), ("my discount code didn't work", "promo_codes"),
    ("how long is the warranty", "warranty"), ("the item i bought is defective", "warranty"), ("it broke after two days", "warranty"),
    ("they sent the wrong colour", "wrong_or_missing_item"), ("my order came without the manual", "wrong_or_missing_item"), ("i got two of the same thing instead", "wrong_or_missing_item"),
    ("my package is late", "delayed_delivery"), ("it's been two weeks and still no delivery", "delayed_delivery"), ("the courier hasn't delivered yet", "delayed_delivery"),
    ("need a bill for my records", "invoice"), ("how can i get my invoice", "invoice"), ("do you have this in stock", "stock_availability"),
    ("when will you have it again", "stock_availability"), ("i need to alter my order", "modify_order"), ("change the quantity of my item", "modify_order"),
    ("i have a gift card, how do i use it", "gift_cards"), ("buy a gift certificate", "gift_cards"),
    ("what's your support number", "contact_support"), ("can i email you", "contact_support"), ("what time do you open", "business_hours"),
    ("are you working on saturday", "business_hours"), ("i want to speak with a representative", "human_agent"), ("transfer me to customer support", "human_agent"),
    ("is this a robot", "bot_identity"), ("who am i talking to", "bot_identity"), ("hi", "greeting"), ("good afternoon", "greeting"),
    ("thank you", "thanks"), ("much appreciated", "thanks"), ("bye bye", "goodbye"), ("see you later", "goodbye"),
    ("wheres my order", "order_status"), ("cancl order", "cancel_order"), ("reund status", "refund_status"), ("deliverry time", "shipping_info"),
    ("pament options", "payment_methods"), ("i cant login", "reset_password"),
]

# off-topic questions: the right behaviour is to say "I don't know" (fallback), not to guess a topic
OFF_TOPIC = [
    "what's the weather today", "tell me a joke", "who won the cricket match", "can you recommend a pizza place", "what is the capital of france",
    "i love pizza", "my dog is sick", "how do i bake a cake", "what is 2 plus 2", "play some music", "who is the prime minister",
    "the moon is made of cheese", "banana pancakes", "xyzzy qwerty",
]

UNSEEN_SCENARIOS = {
    "refund then delivery then refund again": [("what is your refund policy", "refund_policy"), ("how long does delivery take", "shipping_info"),
                                                ("and the refund, how long", "refund_status")],
    "payment then shipping cost": [("what payment methods do you take", "payment_methods"), ("what are the shipping charges", "shipping_info"), ("is it free", "followup_cost", "free over")],
    "track then cancel flow": [("where is order 10002", "order_status"), ("cancel it", "cancel_order"), ("yes", "cancel_order", "Done")],
    "warranty and agent": [("the item is defective", "warranty"), ("is it free", "followup_cost", "free"), ("i want to speak to a human", "human_agent")],
    "return and cost": [("how do i return something", "return_process"), ("any charges", "followup_cost", "pickup fee")],
}
