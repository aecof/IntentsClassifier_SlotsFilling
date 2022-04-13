
import pandas as pd
from src.models import VanillaEncoderDecoder, IntentSlotsClassifier
from src.utils import preprocess_one_sentence
import ray
from ray import serve
import numpy as np
import json
import torch


@serve.deployment(route_prefix="/nlu", name="nlu-deployment")
class NLUDeployment:
    def __init__(self):

        self.embedding_matrix = np.load('src/embedding_matrix.npy')
        self.embedding_matrix = np.zeros(self.embedding_matrix.shape)

        self.slots_dict = pd.read_csv(
            'data_dir/atis.slots.dict.txt', header=None)[0].to_dict()
        self.intents_dict = pd.read_csv(
            'data_dir/atis.intent.dict.txt', header=None)[0].to_dict()
        with open('src/stoi.json', 'r') as f:
            self.stoi = json.load(f)

        self.transformer = preprocess_one_sentence

        self.classifier = VanillaEncoderDecoder(
            self.embedding_matrix.shape[0], self.embedding_matrix.shape[1], self.embedding_matrix, 128, len(self.intents_dict), len(self.slots_dict), n_layers=1).to('cpu')

        self.classifier.load_state_dict(torch.load('src/save/train/server_model-01/best.pt', map_location=torch.device('cpu'))['state_dict']
                                        )
        self.classifier.eval()

    async def __call__(self, request):
        data = await request.body()
        transformed = self.transformer('BOS ' + str(data) + ' EOS', self.stoi)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        transformed = torch.IntTensor(transformed).unsqueeze(dim=0).to(device)
        intent, slots = self.classifier(transformed)
        _, predictions_intent = torch.max(intent.data, 1)
        _, predictions_slots = torch.max(slots.data, 1)
        lol = [self.slots_dict[int(elm)] for elm in predictions_slots[0][1:-1]]
        return f'{self.intents_dict[int(predictions_intent)]}///{lol}'


@serve.deployment(route_prefix="/nlu-joint", name="nlu-joint-deployment")
class NLUJointDeployment:
    def __init__(self):

        self.embedding_matrix = np.load('src/embedding_matrix.npy')
        self.embedding_matrix = np.zeros(self.embedding_matrix.shape)
        self.slots_dict = pd.read_csv(
            'data_dir/atis.slots.dict.txt', header=None)[0].to_dict()
        self.intents_dict = pd.read_csv(
            'data_dir/atis.intent.dict.txt', header=None)[0].to_dict()
        with open('src/stoi.json', 'r') as f:
            self.stoi = json.load(f)

        self.transformer = preprocess_one_sentence

        self.classifier = IntentSlotsClassifier(
            self.embedding_matrix.shape[0], len(self.intents_dict), len(self.slots_dict),  self.embedding_matrix.shape[1], self.embedding_matrix, 128, num_layers=3).to('cpu')

        self.classifier.load_state_dict(torch.load('src/save/train/server_joint_model-01/best.pt', map_location=torch.device('cpu'))['state_dict']
                                        )
        self.classifier.eval()

    async def __call__(self, request):
        data = await request.body()
        transformed = self.transformer('BOS ' + str(data) + ' EOS', self.stoi)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        transformed = torch.IntTensor(transformed).unsqueeze(dim=0).to(device)
        intent, slots = self.classifier(transformed)
        _, predictions_intent = torch.max(intent.data, 1)
        _, predictions_slots = torch.max(slots.data, 1)

        lol = [self.slots_dict[int(elm)] for elm in predictions_slots[0][1:-1]]
        return f'{self.intents_dict[int(predictions_intent)]}///{lol}'


@serve.deployment(route_prefix="/nlu-sep", name="nlu-sep-deployment")
class NLUSeparateDeployment:
    def __init__(self):

        self.embedding_matrix = np.load('src/embedding_matrix.npy')
        self.embedding_matrix = np.zeros(self.embedding_matrix.shape)
        self.slots_dict = pd.read_csv(
            'data_dir/atis.slots.dict.txt', header=None)[0].to_dict()
        self.intents_dict = pd.read_csv(
            'data_dir/atis.intent.dict.txt', header=None)[0].to_dict()
        with open('src/stoi.json', 'r') as f:
            self.stoi = json.load(f)

        self.transformer = preprocess_one_sentence

        self.classifier = IntentSlotsClassifier(
            self.embedding_matrix.shape[0], len(self.intents_dict), len(self.slots_dict),  self.embedding_matrix.shape[1], self.embedding_matrix, 128, num_layers=3).to('cpu')

        self.classifier.load_state_dict(torch.load('src/save/train/server_joint_model-01/best.pt', map_location=torch.device('cpu'))['state_dict']
                                        )
        self.classifier.eval()

    async def __call__(self, request):
        data = await request.body()
        transformed = self.transformer('BOS ' + str(data) + ' EOS', self.stoi)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        transformed = torch.IntTensor(transformed).unsqueeze(dim=0).to(device)
        intent, slots = self.classifier(transformed)
        _, predictions_intent = torch.max(intent.data, 1)
        _, predictions_slots = torch.max(slots.data, 1)

        lol = [self.slots_dict[int(elm)] for elm in predictions_slots[0][1:-1]]
        return f'{self.intents_dict[int(predictions_intent)]}///{lol}'


def main() : 

    # Connect to the running Ray Serve instance.
    ray.init(address='auto', namespace="serve-example",
            ignore_reinit_error=True)
    serve.start(detached=True)

    # Deploy the model.
    NLUDeployment.deploy()
    NLUJointDeployment.deploy()

if __name__=='__main__':
    main()