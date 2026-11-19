import unittest
import torch
from trainer.unlearn.seul import SeUL, select_spans, span_intervals, label_logprobs, selective_ga_loss
from trainer.masking import eligible_positions


class SeULTests(unittest.TestCase):
    def test_mean_strict_and_gap(self):
        s=torch.tensor([[0.,-5.,-1.,-5.,0.,-1.,0.]])
        v=torch.ones_like(s,dtype=torch.bool)
        m,seeds,t=select_spans(s,v)
        self.assertEqual(seeds.nonzero()[:,1].tolist(),[1,3])
        self.assertEqual(m.nonzero()[:,1].tolist(),[1,2,3])
        self.assertEqual(span_intervals(m[0].tolist()),[[1,4]])
        self.assertAlmostEqual(t.item(),s.mean().item())
        s[0,5]=-2.
        m,seeds,t=select_spans(s,v)
        self.assertEqual(seeds.nonzero()[:,1].tolist(),[1,3,5])
        self.assertEqual(m.nonzero()[:,1].tolist(),[1,2,3,4,5])

    def test_ties_and_empty(self):
        s=torch.full((2,4),-2.);v=torch.tensor([[1,1,1,1],[0,0,0,0]],dtype=torch.bool)
        m,_,_=select_spans(s,v);self.assertFalse(m.any())

    def test_fixed_gap_no_cascade(self):
        s=torch.tensor([[-5.,-1.,-1.,-5.,-1.,-5.]])
        m,_,_=select_spans(s,torch.ones_like(s,dtype=torch.bool),'fixed',-4.)
        self.assertEqual(m[0].tolist(),[True,False,False,True,True,True])

    def test_never_fill_ineligible(self):
        s=torch.tensor([[-5.,-1.,-5.]])
        m,_,_=select_spans(s,torch.tensor([[1,0,1]],dtype=torch.bool),'fixed',-4.)
        self.assertFalse(m[0,1])

    def test_padding_invariance(self):
        s=torch.tensor([[-5.,-1.,-5.]])
        v=torch.ones_like(s,dtype=torch.bool)
        expected=select_spans(s,v)[0]
        for left,right in [(0,3),(2,0),(1,2)]:
            ps=torch.nn.functional.pad(s,(left,right),value=float('nan'))
            pv=torch.nn.functional.pad(v,(left,right),value=False)
            actual=select_spans(ps,pv)[0][:,left:left+3]
            self.assertTrue(torch.equal(expected,actual))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):select_spans(torch.tensor([[float('nan')]]),torch.ones((1,1),dtype=torch.bool))
        with self.assertRaises(ValueError):select_spans(torch.zeros((1,2)),torch.ones((1,2),dtype=torch.bool),'fixed')

    def test_shift_and_sum_not_token_mean(self):
        logits=torch.tensor([[[1.,2.,3.],[2.,3.,1.],[3.,1.,2.]]],requires_grad=True)
        labels=torch.tensor([[-100,1,2]]);mask=torch.tensor([[0,1,1]],dtype=torch.bool)
        loss=selective_ga_loss(logits,labels,mask,mask)
        expected=logits[0,0].log_softmax(-1)[1]+logits[0,1].log_softmax(-1)[2]
        self.assertTrue(torch.allclose(loss,expected))
        loss.backward();self.assertEqual(logits.grad[0,2].abs().sum().item(),0.)

    def test_gradient_ascent_direction(self):
        logits=torch.zeros((1,2,3),requires_grad=True);labels=torch.tensor([[-100,1]])
        mask=torch.tensor([[0,1]],dtype=torch.bool)
        before=logits[0,0].softmax(-1)[1].item()
        selective_ga_loss(logits,labels,mask,mask).backward()
        after=(logits-.1*logits.grad)[0,0].softmax(-1)[1].item()
        self.assertLess(after,before)

    def test_empty_graph_zero(self):
        logits=torch.randn((2,3,4),requires_grad=True);labels=torch.full((2,3),-100)
        v=torch.zeros_like(labels,dtype=torch.bool)
        loss=selective_ga_loss(logits,labels,v,v);self.assertTrue(torch.isfinite(loss));self.assertEqual(loss.item(),0.)
        loss.backward();self.assertEqual(logits.grad.abs().sum().item(),0.)

    def test_answer_eos_and_padding(self):
        inputs={'labels':torch.tensor([[-100,3,4,4,-100]]),'attention_mask':torch.tensor([[1,1,1,1,0]])}
        self.assertEqual(eligible_positions(inputs)[0].tolist(),[False,True,True,True,False])

    def test_original_model_frozen_and_no_retain_term(self):
        import tempfile
        from transformers import LlamaConfig,LlamaForCausalLM,TrainingArguments
        torch.manual_seed(3)
        model=LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=16,
            intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
        with tempfile.TemporaryDirectory() as output:
            args=TrainingArguments(output_dir=output,use_cpu=True,report_to=[],remove_unused_columns=False)
            trainer=SeUL(model=model,args=args)
            forget={'input_ids':torch.tensor([[1,2,3,4,5]]),'labels':torch.tensor([[-100,-100,3,4,5]]),
                    'attention_mask':torch.ones((1,5),dtype=torch.long),'index':torch.tensor([0])}
            before=trainer.original_selection(forget).clone()
            reference=[p.clone() for p in trainer.selection_model.parameters()]
            loss=trainer.compute_loss(model,{'forget':forget,'retain':object()})
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:p.add_(p.grad,alpha=-.01)
            self.assertTrue(torch.equal(before,trainer.original_selection(forget,verify_original=True)))
            padded={k:(torch.nn.functional.pad(v,(2,0),value=-100 if k=='labels' else 0)
                       if k!='index' else v) for k,v in forget.items()}
            self.assertTrue(torch.equal(before,trainer.original_selection(padded)[:,2:]))
            self.assertTrue(all(torch.equal(a,b) and not b.requires_grad and b.grad is None
                                for a,b in zip(reference,trainer.selection_model.parameters())))
            other={**forget,'input_ids':forget['input_ids'].clone()};other['input_ids'][0,1]=7
            with self.assertRaises(ValueError):trainer.original_selection(other)

if __name__=='__main__':unittest.main()
