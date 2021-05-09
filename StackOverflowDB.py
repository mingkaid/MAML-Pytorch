import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
from xml.etree import ElementTree as ET

class StackOverflowDB():
    def __init__(self, num_support=1):
        
        assert num_support < 10, "Too many supports, must fewer than 10"
        self.data = []
        self.num_support = num_support
        self.user_idx = []

        classes = ["3d", "health", "cg", "ai", "coffee"]
        for cls in classes:
            self.data.append(defaultdict(list))

            tree = ET.parse(f'/home/mingkaid/github-repos/MAML-Pytorch/MAML-Pytorch/data/{cls}_Comments.xml')
            root = tree.getroot()
            cls_data = []
            user_dict = defaultdict(int)
            for child in root:
                if "UserId" in child.attrib.keys() and "Text" in child.attrib.keys():
                    user_dict[child.attrib["UserId"]] += 1

            filtered_users = []
            for k, v in user_dict.items():
                if v > 10 and v <50:
                    filtered_users.append(k)
            
            self.user_idx.append(filtered_users)
            print(f"number of filtered users for class {cls} is {len(filtered_users)}")
            
            self.full_text = []
            for child in root:
                if "UserId" in child.attrib.keys() and child.attrib["UserId"] in filtered_users\
                    and "Text" in child.attrib.keys():
                    self.full_text.append(child.attrib["Text"])
                    self.data[-1][child.attrib["UserId"]].append(child.attrib["Text"])
            
            self.vocab = dict()
            self.vocab_size = 1000
            self.max_length = 20
            self.build_vocab()
        
    def build_vocab(self):
        vocab_count = defaultdict(int)

        # process vocaulary
        for l in self.full_text:
            for i in l.split():
                vocab_count[i] += 1
        sorted_dict = dict(sorted(vocab_count.items(), key=lambda item: item[1]))
        self.vocab["UNK"] = 0
        for l in range(1, self.vocab_size):
            self.vocab[list(sorted_dict.keys())[-l]] = l
        self.vocab["pad"] = self.vocab_size

    def tokenize(self, l):
        sent = []
        for i in l.split():
            if i in self.vocab:
                idx = self.vocab[i]
            else: # out of vocabulary
                idx = 0
            sent.append(idx)
        while len(sent) <= self.max_length:
            sent.append(self.vocab_size)
        if len(sent) > self.max_length:
            sent = sent[:self.max_length]
        sent = np.asarray(sent)
        return sent
    
    """
        mode: "train", "val", "test"

    """
    def next(self, bs, mode, query_num, query_num_oth):
        if mode == "train":
            cls_idx = np.random.choice(np.array([0,1,2]))
        elif mode == "val":
            cls = 3
        elif mode == "text":
            cls = 4
        else:
            raise ValueError("mode must be in [train, val, test]")

        users_idx = np.random.choice(self.user_idx[cls_idx], size=1+query_num_oth, replace=False)
        user_idx = users_idx[0]
        other_idx = users_idx[1:]

        support = self.data[cls_idx][user_idx][:self.num_support]
        query = self.data[cls_idx][user_idx][self.num_support:]
        oth_query = []
        for i in range(query_num_oth):
            oth_sent_count = len(self.data[cls_idx][other_idx[i]])
            oth_sent_idx = np.random.choice(np.array(list(range(oth_sent_count))))
            oth_query.append(self.data[cls_idx][other_idx[i]][oth_sent_idx])

        support_x = []
        support_y = []
        support_mask = []

        query_x = []
        query_y = []
        query_mask = []

        oth_query_x = []
        oth_query_y = []
        oth_query_mask = []

        for i in range(bs):
            support_sent_idx = np.random.choice(np.asarray(list(range(self.num_support))))
            cur_support = self.tokenize(support[support_sent_idx])

            example_idx = np.random.choice(np.asarray(list(range(self.max_length-1))))
            
            support_x.append(cur_support)
            support_y.append(cur_support[example_idx])

            cur_mask = np.zeros(self.max_length)
            cur_mask[example_idx] = 1

            support_mask.append(cur_mask)

        for i in range(query_num):
            query_sent_idx = np.random.choice(np.asarray(list(range(len(query)))))
            cur_query = self.tokenize(query[query_sent_idx])
          
            example_idx = np.random.choice(np.asarray(list(range(self.max_length-1))))

            query_x.append(cur_query)
            query_y.append(cur_query[example_idx])

            cur_mask = np.zeros(self.max_length)
            cur_mask[example_idx] = 1

            query_mask.append(cur_mask)

        for i in range(len(oth_query)):
            cur_oth_query = self.tokenize(oth_query[i])
              
            example_idx = np.random.choice(np.asarray(list(range(self.max_length-1))))
            
            oth_query_x.append(cur_oth_query)
            oth_query_y.append(cur_oth_query[example_idx])

            cur_mask = np.zeros(self.max_length)
            cur_mask[example_idx] = 1

            oth_query_mask.append(cur_mask)


        support_x = torch.from_numpy(np.asarray(support_x))
        support_y = torch.from_numpy(np.asarray(support_y))
        support_mask = torch.from_numpy(np.asarray(support_mask))

        query_x = torch.from_numpy(np.asarray(query_x))
        query_y = torch.from_numpy(np.asarray(query_y))
        query_mask = torch.from_numpy(np.asarray(query_mask))

        oth_query_x = torch.from_numpy(np.asarray(oth_query_x))
        oth_query_y = torch.from_numpy(np.asarray(oth_query_y))
        oth_query_mask = torch.from_numpy(np.asarray(oth_query_mask))

        return support_x, support_y, support_mask, query_x, query_y, query_mask, oth_query_x, oth_query_y, oth_query_mask