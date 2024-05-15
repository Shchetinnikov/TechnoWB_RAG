import pandas as pd
import torch

from .models import RAG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

knowledge_base = pd.read_excel('server/database/data/knowledge_base.xlsx')
qa_pairs = pd.read_excel('server/database/data/QA_pairs.xlsx')

rag = RAG(knowledge_base, qa_pairs, device)