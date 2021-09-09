from typing import List, Dict
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class NaiveRLDataset(Dataset):

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path
        with open(self._data_path, 'rb') as f:
            self._data: List[Dict[str, torch.Tensor]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

class OfflineRLDataset(Dataset):

    def __init__(self, data_path: str, norm: bool = False, data_transform_type: str = 'identity') -> None:
        self._data_path = data_path
        self._data = torch.load(self._data_path)
        if data_transform_type=='tanh':
            self._data_tanh()
        elif data_transform_type=='clip':
            self._data_clip()
            
        if norm:
            self._norm()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

    def _norm(self):
        obs=[]
        for i in tqdm(range(len(self._data))):
            obs.append(self._data[i]['obs'])
        obs = torch.stack(obs, dim=0)
        mu = torch.mean(obs,dim=0,keepdim=True)
        sd = torch.std(obs,dim=0,keepdim=True)
        normalized_obs = (obs - mu)/(sd+1e-7)
        for i in tqdm(range(len(self._data))):
            self._data[i]['obs']=normalized_obs[i]
    
    def _data_tanh(self):
        for i in tqdm(range(len(self._data))):
            self._data[i]['action']=torch.tanh(self._data[i]['action'])
    
    def _data_clip(self):
        for i in tqdm(range(len(self._data))):
            self._data[i]['action']=torch.clamp(self._data[i]['action'],min=-1,max=1)

class D4RLDataset(Dataset):
    
    def __init__(self, env_id: str, device: str) -> None:
        import gym
        import d4rl # Import required to register environments
        # Create the environment
        self._device = device
        env = gym.make(env_id)
        if 'random-expert' in env_id:
            dataset = d4rl.basic_dataset(env)
        else:
            dataset = d4rl.qlearning_dataset(env)
        self._data = []
        self._load_d4rl(dataset)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

    def _load_d4rl(self, dataset):
        for i in range(len(dataset['observations'])):
            # if i >1000:
            #     break
            trans_data={}
            # import ipdb;ipdb.set_trace()
            trans_data['obs'] = torch.from_numpy(dataset['observations'][i])
            trans_data['next_obs'] = torch.from_numpy(dataset['next_observations'][i])
            trans_data['action'] = torch.from_numpy(dataset['actions'][i])
            trans_data['reward'] = torch.tensor(dataset['rewards'][i])
            trans_data['done'] = dataset['terminals'][i]
            trans_data['collect_iter'] = 0
            self._data.append(trans_data)