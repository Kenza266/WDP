from cmath import e
import streamlit as st
import numpy as np
from numpy import savetxt
from numpy import genfromtxt
import random
import operator
from math import ceil
from stqdm.stqdm import stqdm
import matplotlib.pyplot as plt
import time
import warnings
from io import BytesIO
import pandas as pd
warnings.filterwarnings('ignore')

gains, winners = [], []
solvers = ['Brute force','Q-Learning', 'Tabu Search']
my_color = 'red'

if 'instance' not in st.session_state:
	st.session_state.instance = None
if 'styler' not in st.session_state:
	st.session_state.styler = None
if 'styler_demo' not in st.session_state:
	st.session_state.styler_demo = None
if 'report' not in st.session_state:
	st.session_state.report = None
if 'report_demo' not in st.session_state:
	st.session_state.report_demo = None
if 'solver' not in st.session_state:
	st.session_state.solver = None

class Instance():
  class bidder:
    def __init__(self, items, price):
      self.items = items
      self.price = price

  def __init__(self, num_items, num_bidders, min_items, max_items, max_price):
    self.items = np.arange(num_items)
    self.bidders = np.array([]) 
    
    for i in range(num_bidders):
      self.add(np.unique(np.random.randint(0, num_items, random.randint(min_items, max_items))), random.randint(max_price/2, max_price), self.items) 
    
    self.available_items = np.arange(num_items)
    self.rem_bidders = np.arange(num_bidders)
  
  def add(self, items, price, all_items):
    items = np.array(items)
    if(all(x in all_items for x in items)):
      self.bidders = np.append(self.bidders, self.bidder(items, price))

  def possible_bidders(self):
    poss_bidders = []
    for b in self.bidders:
      if(all(x in self.available_items for x in b.items)):
        poss_bidders.append(np.where(self.bidders == b)[0][0])
    return poss_bidders

  def choose_bidder(self):
    return np.random.choice(self.possible_bidders(), 1)[0]

  def choose_best_bidder_Q(self):
    return self.Q.sum(axis=0).argmax()

  def update(self, bidder, prev_bidder):
    max_index = np.where(self.Q[prev_bidder, ] == np.max(self.Q[prev_bidder, ]))[1]
    if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
    else:
      max_index = int(max_index)
    max_value = self.Q[prev_bidder, max_index] 

    self.Q[prev_bidder, bidder] = self.bidders[bidder].price + self.gamma * max_value

    if (np.max(self.Q) > 0):
      return np.sum(self.Q / np.max(self.Q)*100)
    else:
      return 0

  def train(self, epochs, early=True, gamma=0.4, e=1e-8, stable=10):
    self.Q = np.matrix(np.zeros([len(self.bidders), len(self.bidders)])) 
    self.gamma = gamma
    scores = []
    for i in stqdm(range(epochs), desc="Training of the Q table", colour=my_color): 
      sum = 0
      self.available_items = self.items
      self.rem_bidders = np.arange(len(self.bidders))
      bidder = self.choose_bidder()
      scores.append(np.sum(self.Q / np.max(self.Q)*100) if (np.max(self.Q) > 0) else 0) 
      prev_bidder = bidder 
      self.rem_bidders = np.array([b for b in self.rem_bidders if b != bidder])
      self.available_items = np.array([i for i in self.available_items if i not in self.bidders[bidder].items])
      while(len(self.possible_bidders()) > 0):
        bidder = self.choose_bidder()
        score = self.update(bidder, prev_bidder)
        prev_bidder = bidder
        self.rem_bidders = np.array([b for b in self.rem_bidders if b != bidder])
        self.available_items = np.array([i for i in self.available_items if i not in self.bidders[bidder].items])
        if(early and self.early_stop(scores, e, ceil(epochs/stable))):
          return scores
        scores.append(score)    
    return scores

  def early_stop(self, scores, e, stable=50):
    if(len(scores) < stable):
      return False
    return ((scores[-stable] - scores[-1])**2 < e)

  def load(self, q_matrix):
    if(len(self.bidders) <= q_matrix.shape[0]):
      self.Q = q_matrix
    else:
      print("Max number of bidders reached")

  def evaluate(self):
    sum = 0
    winners = []
    self.available_items = self.items
    self.rem_bidders = np.arange(len(self.bidders))

    bidder = self.choose_best_bidder_Q()
    winners.append(bidder)
    prev_bidder = bidder
    sum += self.bidders[bidder].price 

    self.rem_bidders = np.array([b for b in self.rem_bidders if b != bidder])
    self.available_items = np.array([i for i in self.available_items if i not in self.bidders[bidder].items])

    while (len(self.possible_bidders()) > 0):
      possible_bidders_scores = {b: self.Q[prev_bidder, b] for b in self.possible_bidders()}
      bidder = max(possible_bidders_scores.items(), key=operator.itemgetter(1))[0]
      winners.append(bidder)
      prev_bidder = bidder
      sum += self.bidders[bidder].price
      self.rem_bidders = np.array([b for b in self.rem_bidders if b != bidder])
      self.available_items = np.array([i for i in self.available_items if i not in self.bidders[bidder].items])
    return winners, sum

  def run(self, epochs=1000, stable=100, early=10):
    history = [] 
    tl = []
    best_solution = [[], 0]
    for i in stqdm(range(epochs), desc="Tabu search", colour=my_color): 
      self.available_items = self.items
      self.rem_bidders = np.arange(len(self.bidders))
      solution = [[], 0]
      possible = [b for b in self.possible_bidders() if solution[0]+[b] not in tl]
      possible = sorted(possible, key=lambda x: self.bidders[x].price, reverse=True) 
      while(len(possible) != 0):
        bidder = possible[0] 
        self.rem_bidders = np.array([b for b in self.rem_bidders if b != bidder])
        self.available_items = np.array([i for i in self.available_items if i not in self.bidders[bidder].items])
        solution[0].append(bidder)
        solution[1] += self.bidders[bidder].price
        history.append(best_solution[1])
        tl.append(solution[0])
        if(solution[1] > best_solution[1]):
          best_solution = solution.copy()
          best_i = i
        if(self.early_stop(history, ceil(epochs/stable))): 
          self.available_items = self.items
          self.rem_bidders = np.arange(len(self.bidders))
          possible = [b for b in self.possible_bidders() if solution[0]+[b] not in tl]
          possible = sorted(possible, key=lambda x: self.bidders[x].price, reverse=True) 
          bidder = possible[0] 
          self.rem_bidders = np.array([b for b in self.rem_bidders if b != bidder])
          self.available_items = np.array([i for i in self.available_items if i not in self.bidders[bidder].items])
          solution = [[bidder], self.bidders[bidder].price]
          tl.append(solution[0])
        possible = [b for b in self.possible_bidders() if solution[0]+[b] not in tl]
        possible = sorted(possible, key=lambda x: self.bidders[x].price, reverse=True) 
      if(self.early_stop(history, ceil(epochs/early))):
        return history, best_solution 
    return history, best_solution

def brute_force(sum, winner, av_bidders, av_items):
  for b in av_bidders:
    if (all(x in av_items for x in b.items)):
      brute_force(sum + b.price, winner + 1,
                  np.array([i for i in av_bidders if i != b]), 
                  np.array([i for i in av_items if i not in b.items]))
  gains.append(sum)
  winners.append(winner)

def WDP(av_bidders, av_items): 
  brute_force(0, 0, av_bidders, av_items) 
  return np.max(np.array(winners)[np.where(np.array(gains) == np.max(np.array(gains)))]), np.max(np.array(gains))

def init_instance():
    st.session_state.instance = None

st.title("Winner Determination Problem")

st.header("Number of bids")
nb_bids = st.number_input('Please select the number of bids', 2, 5000, on_change=init_instance)
st.header("Number of items")
nb_items = st.number_input('Please select the number of items', 1, 5000, on_change=init_instance)
st.header("Number of items per bid")
min_b, max_b = (st.number_input('Min', 1, nb_items, on_change=init_instance), 
                st.number_input('Max', 1, nb_items, on_change=init_instance))

demo_button = st.button('Demo')
if demo_button:
    if st.session_state.instance == None:
        st.session_state.instance = Instance(nb_items, nb_bids, min_b, max_b, max_price=2000) 

    start_time = time.time() 
    scores = st.session_state.instance.train(5000)
    w_Q, g_Q = st.session_state.instance.evaluate()
    t_Q = time.time() - start_time

    start_time = time.time() 
    history, solution = st.session_state.instance.run(epochs=1000, stable=10, early=10)
    t_TS = time.time() - start_time

    gains, winners = [], []
    info = st.info("Running the brute force...")
    start_time = time.time() 
    w_BF, g_BF = WDP(st.session_state.instance.bidders, st.session_state.instance.items) 
    t_BF = time.time() - start_time
    info.empty()
    df_demo = pd.DataFrame({'Solver':solvers, 'Time(s)':[t_BF, t_Q, t_TS], 
                            'Winning bids':[w_BF, len(w_Q), len(solution[0])],
                            'Gain':[g_BF, g_Q, solution[1]]})
    st.session_state.styler_demo = df_demo.style.hide_index().format(subset=['Time(s)'], decimal=',', precision=2).bar(subset=['Time(s)'], align="mid", color=my_color)
    st.session_state.report_demo = st.write(st.session_state.styler_demo.to_html(), unsafe_allow_html=True)
elif st.session_state.styler_demo:
    st.session_state.report_demo = st.write(st.session_state.styler_demo.to_html(), unsafe_allow_html=True)
st.header("With a solver")
solver = st.radio('Please select a solver', solvers)
run_button = st.button('Run')

if run_button:
    if st.session_state.instance == None:
        st.session_state.instance = Instance(nb_items, nb_bids, min_b, max_b, max_price=2000) 
    st.session_state.solver = solver
    st.subheader(st.session_state.solver)
    if st.session_state.solver=='Brute force':
        gains, winners = [], []
        info = st.info("Running the brute force...")
        start_time = time.time() 
        w, g = WDP(st.session_state.instance.bidders, st.session_state.instance.items) 
        t = time.time() - start_time
        info.empty()
        df = pd.DataFrame({'Time(s)':[t], 
                       'Winning bids':[w],
                       'Gain':[g]})
        st.session_state.styler = df.style.hide_index()
        st.session_state.report = st.write(st.session_state.styler.to_html(), unsafe_allow_html=True)
    if st.session_state.solver=='Q-Learning':
        start_time = time.time() 
        scores = st.session_state.instance.train(5000)
        w_Q, g_Q = st.session_state.instance.evaluate()
        t = time.time() - start_time
        w, g = len(w_Q), g_Q
        df = pd.DataFrame({'Time(s)':[t], 
                       'Winning bids':[w],
                       'Gain':[g]})
        st.session_state.styler = df.style.hide_index()
        st.session_state.report = st.write(st.session_state.styler.to_html(), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(scores, color='red') 
        ax.set_xlabel('No of iterations') 
        ax.set_ylabel('Reward gained')
        ax.set_title('Evolution of the reward gained')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
    if st.session_state.solver=='Tabu Search':
        start_time = time.time() 
        history, solution = st.session_state.instance.run(epochs=1000, stable=10, early=10)
        t = time.time() - start_time
        w, g = len(solution[0]), solution[1]
        df = pd.DataFrame({'Time(s)':[t], 
                       'Winning bids':[w],
                       'Gain':[g]})
        st.session_state.styler = df.style.hide_index()
        st.session_state.report = st.write(st.session_state.styler.to_html(), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history, color='red') 
        ax.set_xlabel('No of iterations') 
        ax.set_ylabel('Reward gained')
        ax.set_title('Evolution of the reward gained')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
elif st.session_state.styler:
    st.subheader(st.session_state.solver)
    st.session_state.report = st.write(st.session_state.styler.to_html(), unsafe_allow_html=True)