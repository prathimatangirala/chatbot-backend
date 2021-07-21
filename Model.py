import collections

import torch
from pytorch_transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils import get_answer, input_to_squad_example, squad_examples_to_features

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class QA:

    def __init__(self):
        self.maxLength = 384
        self.docStride = 128
        self.do_lower_case = True
        self.queryLength = 64
        self.answerLength = 30
        self.model, self.tokenizer = self.load_model("output_model")
        self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_model(model_path):
        config = BertConfig.from_pretrained(model_path + "/bert_config.json")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
        return model, tokenizer

    def predict(self, passage, question):
        example = input_to_squad_example(passage, question)
        features = squad_examples_to_features(example, self.tokenizer, self.maxLength, self.docStride,
                                              self.queryLength)
        inputIds = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(inputIds, torch.tensor([f.input_mask for f in features], dtype=torch.long),
                                torch.tensor([f.segment_ids for f in features], dtype=torch.long),
                                torch.arange(inputIds.size(0), dtype=torch.long))
        data_loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=1)
        results = []
        for batch in data_loader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]
                          }
                indices = batch[3]
                outputs = self.model(**inputs)

            for i, example_index in enumerate(indices):
                feature = features[example_index.item()]
                result = RawResult(unique_id=int(feature.unique_id),
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]))
                results.append(result)
        answer = get_answer(example, features, results, 20, self.answerLength, True)
        return answer
