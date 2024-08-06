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
import itertools
import time
import easyocr

pyautogui.PAUSE = 0
reader=easyocr.Reader(["en"])
training_running = False
counter=0
# Define neural network architecture
class OsuManiaNet(nn.Module):
    def __init__(self, input_channels=1, output_size=16):
        super(OsuManiaNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self._to_linear = None
        self.convs(torch.randn(1, input_channels, 140, 50))  # Dummy forward pass to calculate the output size
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, output_size)

    def convs(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        if self._to_linear is None:
            self._to_linear = x.numel()
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.softmax(self.fc2(x), dim=1)
        return x
    
# Define reinforcement learning agent
class OsuManiaAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OsuManiaNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = []
        self.action_map = list(itertools.product([0, 1], repeat=4))
        self.epsilon = -0.5

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = [np.random.randint(2),np.random.randint(2),np.random.randint(2),np.random.randint(2)]  # Random action
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
                action_probs = self.model(state)
                action_index = torch.multinomial(action_probs, 1).item()
                #action_index = torch.argmax(action_probs)
            action = self.action_map[action_index]
        return list(action)
    
    def output_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
            action_probs = self.model(state)
            print(action_probs)
            action_index = torch.argmax(action_probs)
        action = self.action_map[action_index]
        return list(action)

    def update_model(self, low, up):
        if up>len(self.replay_buffer):
            up=len(self.replay_buffer)
        if low>=len(self.replay_buffer):
            return
        batch=self.replay_buffer[low:up]
        states, actions, rewards = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        action_indices = torch.tensor([self.action_map.index(tuple(action)) for action in actions], dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        probs = self.model(states)
        gathered_probs = probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        def loss_func(probs, rewards):
            loss_terms = []
            for prob, reward in zip(probs, rewards):
                if reward <0.5:
                    loss_term = (torch.exp(prob)-1) * 3
                else:
                    loss_term = -torch.log(prob) * reward

                loss_terms.append(loss_term)
            return torch.mean(torch.stack(loss_terms))
        
        loss_val = loss_func(gathered_probs, rewards)
        print(loss_val)
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()

    def train(self, num_episodes, env):
        global counter
        #self.epsilon-=0.1
        for episode in range(num_episodes):
            counter=0
            start=time.time()
            print("episode:",episode)
            if not training_running:
                break
            state = env.reset()
            done = False
            while not done and training_running:
                action = self.select_action(state)
                #action = self.output_action(state)
                state, done = env.step(action)
                if done:
                    break
        
            if done:
                end=time.time()
                print(end-start)
                print(counter)
                self.extract_data(env)
                if counter/(end-start)>=27:
                    for i in range(6):
                        self.update_model(i*512,(i+1)*512)
                self.replay_buffer.clear()
                self.save_model(f'Osu_ai/model/reinforcement_model_{episode}_{env.accuracy}.pth')
                pyautogui.click(1627,814)
                time.sleep(2)

    def extract_data(self, env):
        for i in range(1,len(env.record)):
            reward = env.calculate_reward(env.record[i][1], env.record[i][2], env.record[i][3], env.record[i][4])
            """
            cv2.imshow('f',env.record[i-1][0].squeeze(0))
            cv2.waitKey(0)
            print(env.record[i][3])
            print(reward)
            """
            if reward == None:
                continue
            self.replay_buffer.append((env.record[i-1][0], env.record[i][4], reward))

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
        self.observation_space = spaces.Box(low=0, high=1, shape=(140, 50, 1), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2]) 
        self.accuracy = 0
        self.combo = 0
        self.keys = ['d', 'f', 'j', 'k']
        self.record=[]
        self.fixed_accuracy= {0:0, 50:1.5, 100:2, 200:2.5, 300:6}

    def reset(self):
        self.accuracy = 100  
        self.combo=0
        self.record.clear()
        pyautogui.keyUp('d')
        pyautogui.keyUp('f')
        pyautogui.keyUp('j')
        pyautogui.keyUp('k')
        with mss.mss() as sct:
            rect=sct.monitors[0]
            img = np.array(sct.grab(rect))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        observation = self.preprocess_screen(img[825:925, 305:445],img[825:925, 620:755])
        return observation

    def step(self, action):
        global counter
        counter+=1
        self.perform_action(action)
        with mss.mss() as sct:
            rect=sct.monitors[0]
            img = np.array(sct.grab(rect))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        next_state_observation = self.preprocess_screen(img[825:925, 305:445],img[825:925, 620:760])
        self.record.append([next_state_observation,img[65:110, 1740:1865], img[965:1060, 440:622], img[390:490,430:630], action])
        done = self.is_episode_done(img[0:140, 1320:1920])

        return next_state_observation, done

    def preprocess_screen(self, left_img, right_img):
        img = cv2.hconcat([left_img, right_img])
        img = cv2.resize(img, (140, 50))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def perform_action(self, action):
        for i, key_action in enumerate(action):
            if key_action == 1:  # Key down
                pyautogui.keyDown(self.keys[i])
            elif key_action == 0:  # Key up
                pyautogui.keyUp(self.keys[i])


    def calculate_reward(self, img1, img2, img3, action):
        new_accuracy = self.detect_accuracy(img1)
        new_combo = self.detect_combo(img2)
        if new_accuracy==None or new_combo==None:
            return None
        reward=0
        if new_combo > self.combo:
            fixed_accuracy=self.detect_fixed_accuracy(img3)
            if new_accuracy == self.accuracy:
                reward=96
                reward-=(action.count(1)-(new_combo-self.combo))*48
            elif new_accuracy > self.accuracy:
                if fixed_accuracy!=0:
                    reward=fixed_accuracy
                    reward-=(action.count(1)-(new_combo-self.combo))*1.5
            else:
                reward=fixed_accuracy
                if reward==6:
                    reward=0.5
                
        
        elif new_combo == self.combo:
            if new_accuracy == self.accuracy and action==[0,0,0,0] and self.accuracy!=0:
                reward=0.5

        self.combo = new_combo
        self.accuracy = new_accuracy
        return reward

    def is_episode_done(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        accuracy_text=reader.readtext(gray,detail=0,allowlist='0123456789.')
        """
        cv2.imshow("f",gray)
        cv2.waitKey(0)
        print(accuracy_text)
        """
        try:
            accuracy = float(accuracy_text[0])
            if accuracy>100 or accuracy<0:
                accuracy=None
        except:
            accuracy = None 
        return accuracy

    def detect_combo(self, img):
        # Use OCR to read accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        combo_text=reader.readtext(gray,detail=0,allowlist='0123456789',mag_ratio=1.4)
        try:
            combo = int(combo_text[0])
            if combo-self.combo>=5:
                combo=self.combo-1
            if combo<0:
                combo = None
        except:
            combo = 0 
        return combo
    
    def detect_fixed_accuracy(self, img):
        # Use OCR to read accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fixed_accuracy_text=reader.readtext(gray,detail=0,allowlist='01235')
        try:
            fixed_accuracy = int(fixed_accuracy_text[0])
            if fixed_accuracy not in self.fixed_accuracy.keys():
                fixed_accuracy = 0
        except:
            fixed_accuracy = 0 
        return self.fixed_accuracy[fixed_accuracy]
    

def train_agent(agent, num_episodes):
    env = OsuManiaEnv()
    if agent==None:
        agent = OsuManiaAgent()
    agent.train(num_episodes, env)
