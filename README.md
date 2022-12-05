# DADP: Domain-Adapted Dependency Parsing for Cross-Domain Named Entity Recognition

## Download dataset

You need to download each dataset and put it in the data folder.  In the folder data_sample, we just take samples from the ontonotes5 dataset as an example. 

## Run

Two steps are required in DADP.

### Step 1：Train mainly on DP task for DP pretrained model.

```bash
bash scripts/pretrain_on_dp.sh
```

- The training log and model are saved at outputs/pretrained_on_dp_ontonotes/
- If needed, pls feel free to contact us for the pretrained DP model, which can be directly applied for your own NER task.

### Step 2：Train mainly on NER task for NER final model.

```bash
bash scripts/run_ner.sh
```

- Make sure outputs/pretrained_on_dp_ontonotes/pytorch_model.bin exists.
- In the file run_ner.sh, you can flexibly config hyper-parameters for NER tasks. The training log and model are saved at outputs/
