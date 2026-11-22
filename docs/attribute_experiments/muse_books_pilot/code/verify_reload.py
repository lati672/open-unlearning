from pathlib import Path
import sys,json,gc
R=Path('/workspace/open-unlearning/saves/experiments/muse_books_attribute_pilot_v1')
sys.path.insert(0,str(R/'code'))
import run_pilot as p
import torch
import numpy as np
import torch.nn.functional as F
from safetensors.torch import load_file
from data.collators import DataCollatorForSupervisedDataset
p.locked()
for arm in ['attribute','random_attribute']:
 model,tok=p.load();path=R/arm/'checkpoint-4/updated_layers.safetensors'
 meta=p.read(path.parent/'checkpoint_metadata.json');assert p.sha(path)==meta['weights_sha256']
 state=load_file(str(path),device='cpu');keys=set(model.state_dict())
 result=model.load_state_dict(state,strict=False)
 assert not result.unexpected_keys and set(result.missing_keys)==keys-set(state)
 model.eval();records,_=p.inputs();collator=DataCollatorForSupervisedDataset(tok,index='index');saved=np.load(R/arm/'token_logprobs.npz')
 differences={}
 with torch.inference_mode():
  for role in ['forget','retain']:
   doc=next(iter(records[role]));b=p.batch(records[role],[doc],collator)
   out=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False)
   scores=-F.cross_entropy(out.logits[:,:-1].float().transpose(1,2),b['labels'][:,1:],ignore_index=-100,reduction='none')
   scores=F.pad(scores,(1,0),value=0)[0].cpu().numpy();expected=saved[f'{role}_{doc}']
   np.testing.assert_allclose(scores,expected,rtol=1e-6,atol=1e-5)
   differences[role]={'doc_id':doc,'max_abs_logprob_difference':float(np.max(np.abs(scores-expected)))}
   del out,scores,b
 p.save(R/arm/'checkpoint_reload_verified.json',{'original_target_plus_partial_state_reproduces_saved_diagnostics':True,'checks':differences,'optimizer_updates':0,'partial_state_sha256':meta['weights_sha256']})
 print('CHECKPOINT_RELOAD_OK',arm,json.dumps(differences),flush=True)
 del model,tok,state;gc.collect();torch.cuda.empty_cache()
