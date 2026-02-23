import pandas as pd
import numpy as np
from olist.data import Olist

class Order:
    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self):
        orders = self.data['orders'].copy()
        orders = orders[orders['order_status'] == 'delivered'].copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
        
        orders['wait_time'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
        orders['expected_wait_time'] = (orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']).dt.days
        orders['delay_vs_expected'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
        orders['delay_vs_expected'] = orders['delay_vs_expected'].apply(lambda x: x if x > 0 else 0)
        
        return orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected']]

    def get_review_score(self):
        reviews = self.data['order_reviews'].copy()
        return reviews[['order_id', 'review_score']]

    def get_training_data(self, with_distance_seller_customer=False):
        wait_time = self.get_wait_time()
        review_score = self.get_review_score()
        return wait_time.merge(review_score, on='order_id').dropna()
