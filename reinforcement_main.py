import cv2
import numpy as np
import mss
import mss.tools
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from collections import deque
import random
import pytesseract
import time
import easyocr

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
pyautogui.PAUSE = 0
reader=easyocr.Reader(["en"])
training_running = False
counter=0
# Define neural network architecture
class OsuManiaNet(nn.Module):
    def __init__(self, input_channels=1, key_state_size=4, output_size=8):
        super(OsuManiaNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self._to_linear = None
        self.convs(torch.randn(1, input_channels, 140, 70))  # Dummy forward pass to calculate the output size
        self.fc1 = nn.Linear(self._to_linear + key_state_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def convs(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        if self._to_linear is None:
            self._to_linear = x.numel()
        return x

    def forward(self, x, key_states):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, key_states), dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 4, 2)
    
# Define reinforcement learning agent
class OsuManiaAgent:
    def __init__(self, epsilon=0.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OsuManiaNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = epsilon

    def select_action(self, state, key_states):
        if np.random.rand() < self.epsilon:
            action = [np.random.randint(2),np.random.randint(2),np.random.randint(2),np.random.randint(2)]  # Random action
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
                key_states = torch.FloatTensor(key_states).unsqueeze(0).to(self.device)
                q_values = self.model(state, key_states)
                _, action_indices = torch.max(q_values, dim=2)
                action=action_indices.squeeze(0).tolist()
        return action

    def update_model(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        states, key_states, actions, rewards, next_states, next_key_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        key_states = torch.tensor(np.array(key_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        next_key_states = torch.tensor(np.array(next_key_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        # Compute Q-values for current states
        q_values = self.model(states, key_states).gather(2, actions.unsqueeze(2)).squeeze(2)
        # Compute Q-values for next states
        next_q_values, _ = self.model(next_states, next_key_states).max(dim=2)
        next_q_values[dones.bool()] = 0.0  # Set Q-value to 0 for terminal states
        # Compute target Q-values
        target_q_values = rewards + self.gamma * next_q_values
        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes, env):
        global counter
        for episode in range(num_episodes):
            counter=0
            start=time.time()
            self.epsilon=self.epsilon*0.9
            print("episode:",episode)
            if not training_running:
                break
            state, key_states = env.reset()
            done = False
            while not done and training_running:
                action = self.select_action(state, key_states)
                next_state, next_key_states, reward, done = env.step(action)
                while reward is None and not done:
                    next_state, next_key_states, reward, done = env.step(action)
                if done:
                    break
                self.replay_buffer.append((state, key_states, action, reward, next_state, next_key_states, done))
                state = next_state
                key_states = next_key_states
        
            if done:
                end=time.time()
                print(end-start)
                print(counter)
                if counter/(end-start)>=15.5:
                    for _ in range(4):
                        self.update_model(batch_size=512)
                else:
                    self.replay_buffer.clear()
                
                time.sleep(4)
                self.save_model(f'Osu_ai/model/model_episode_{episode}_{env.accuracy}.pth')
                pyautogui.click(1627,814)
                time.sleep(3.25)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model loaded from {path}')

# Define game environment
class OsuManiaEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(140, 70, 1), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2]) 
        self.key_mappings = {0: 'up', 1: 'down'}
        self.accuracy = 0
        self.score = 0
        self.key_states = np.zeros(4)  # (D, F, J, K)
        self.keys = ['d', 'f', 'j', 'k']

    def reset(self):
        self.accuracy = 0  
        self.score=0
        self.key_states = np.zeros(4)
        pyautogui.keyUp('d')
        pyautogui.keyUp('f')
        pyautogui.keyUp('j')
        pyautogui.keyUp('k')
        with mss.mss() as sct:
            rect=sct.monitors[0]
            img = np.array(sct.grab(rect))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        observation = self.preprocess_screen(img)
        return observation, self.key_states

    def step(self, action):
        global counter
        counter+=1
        #start=time.time()
        self.perform_action(action)
        #print("action time:",time.time()-start)
        #start=time.time()
        with mss.mss() as sct:
            rect=sct.monitors[0]
            img = np.array(sct.grab(rect))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        #print("cap time:",time.time()-start)
        #start=time.time()
        observation = self.preprocess_screen(img[784:924, 300:580])
        #print("preprocess time:",time.time()-start)
        #start=time.time()
        reward = self.calculate_reward(img[65:110, 1740:1865], img[980:1050, 375:510], action)
        #print("cal time:",time.time()-start)
        #start=time.time()
        done = self.is_episode_done(img[0:140, 1320:1920])
        #print("done time:",time.time()-start)

        return observation, self.key_states, reward, done

    def preprocess_screen(self, img):
        img = cv2.resize(img, (140, 70))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def perform_action(self, action):
        for i, key_action in enumerate(action):
            if key_action == 1:  # Key down
                pyautogui.keyDown(self.keys[i])
                self.key_states[i] = 1
            elif key_action == 0:  # Key up
                pyautogui.keyUp(self.keys[i])
                self.key_states[i] = 0


    def calculate_reward(self, img1, img2, action):
        new_accuracy = self.detect_accuracy(img1)
        new_score = self.detect_score(img2)
        if new_accuracy==None:
            return None

        if new_accuracy>100 or new_accuracy<0:
            new_accuracy = self.accuracy
        if new_accuracy < self.accuracy:
            reward=-2
        elif new_accuracy > self.accuracy:
            reward=min((new_accuracy - self.accuracy)*counter/15,10)
        elif new_score==self.score and action ==[0,0,0,0]:
            reward=0.5
        else:
            reward=0
        if new_score>self.score:
            reward+=2.5
        elif self.score>new_score:
            reward-=2
        self.score = new_score
        self.accuracy = new_accuracy

        return reward

    def is_episode_done(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('Osu_ai/game_over.png', 0) 
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7  # Adjust this value as needed
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            return True
        else:
            return False

    def detect_accuracy(self, img):
        # Use OCR to read accuracy
        score_text=reader.readtext(img,detail=0,allowlist='0123456789.')
        try:
            score = float(score_text[0])
        except:
            score = None 
        return score
    
    def detect_score(self, img):
        # Use OCR to read accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score_text=reader.readtext(gray,detail=0,allowlist='0123456789')
        try:
            score = int(score_text[0])
        except:
            score = self.score  
        return score
    

def train_agent(agent, num_episodes, epsilon=0.0):
    env = OsuManiaEnv()
    if agent==None:
        agent = OsuManiaAgent(epsilon)
    agent.train(num_episodes, env)

