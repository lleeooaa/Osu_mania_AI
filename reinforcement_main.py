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
        x = self.fc2(x)
        return x
    
# Define reinforcement learning agent
class OsuManiaAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OsuManiaNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=3000)
        self.action_map = list(itertools.product([0, 1], repeat=4))

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
            action_probs = nn.functional.softmax(self.model(state), dim=1)
            action_index = torch.multinomial(action_probs, 1).item()
        action = self.action_map[action_index]
        return list(action)

    def update_model(self, low, up):
        if up>len(self.replay_buffer):
            return
        batch=deque(itertools.islice(self.replay_buffer, low, up))
        states, actions, losses = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        action_indices = torch.tensor([self.action_map.index(tuple(action)) for action in actions], dtype=torch.int64).to(self.device)
        losses = torch.tensor(np.array(losses), dtype=torch.float32).unsqueeze(1).to(self.device)
        # Compute Q-values for current states
        probs = nn.functional.softmax(self.model(states), dim=1)
        gathered_probs = probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Compute the loss as the negative log probability of the actions taken, weighted by the losses
        loss_val = (gathered_probs * losses + 1 - gathered_probs).mean()

        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()

    def train(self, num_episodes, env):
        global counter
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
                        self.update_model(i*500,(i+1)*500)
                else:
                    self.replay_buffer.clear()
                self.save_model(f'Osu_ai/model/reinforcement_episode_{episode}.pth')
                pyautogui.click(1627,814)
                time.sleep(2)

    def extract_data(self, env):
        for i in range(len(env.record)):
            loss = env.calculate_loss(env.record[i][1], env.record[i][2], env.record[i][3])
            if loss == None:
                continue
            self.replay_buffer.append((env.record[i][0], env.record[i][3], loss))

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
        self.combo = 0
        self.keys = ['d', 'f', 'j', 'k']
        self.record=[]

    def reset(self):  
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
        observation = self.preprocess_screen(img)
        return observation

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
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        #print("cap time:",time.time()-start)
        #start=time.time()
        observation = self.preprocess_screen(img[824:924, 300:580])
        self.record.append([observation,img[280:400, 320:550], img[980:1050, 375:510], action])
        #print("preprocess time:",time.time()-start)
        #start=time.time()
        #print("cal time:",time.time()-start)
        #start=time.time()
        done = self.is_episode_done(img[0:140, 1320:1920])
        #print("done time:",time.time()-start)

        return observation, done

    def preprocess_screen(self, img):
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


    def calculate_loss(self, img1, img2, action):
        accuracy = self.detect_accuracy(img1)
        new_combo = self.detect_combo(img2)
        if accuracy==300:
            loss=0
        elif accuracy==200:
            loss=2
        elif accuracy==100:
            loss=4
        elif accuracy==50:
            loss=6 
        elif accuracy==0:
            loss=8
        elif (new_combo==self.combo and action==[0,0,0,0]) or (new_combo>self.combo):
            loss=0
        elif self.combo>new_combo:
            loss=None
        else:
            loss=2
        self.combo = new_combo
        return loss

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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        accuracy_text=reader.readtext(gray,detail=0,allowlist='01235miss!')
        if accuracy_text=="miss!":
            return 0
        try:
            accuracy = int(accuracy_text[0])
        except:
            accuracy = None
        if accuracy!=50 and accuracy!=100 and accuracy!=200 and accuracy!=300:
            accuracy=None
        return accuracy
    
    def detect_combo(self, img):
        # Use OCR to read accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        combo_text=reader.readtext(gray,detail=0,allowlist='0123456789')
        try:
            combo = int(combo_text[0])
        except:
            combo = self.combo  
        return combo
    

def train_agent(agent, num_episodes):
    env = OsuManiaEnv()
    if agent==None:
        agent = OsuManiaAgent()
    agent.train(num_episodes, env)

