# EADSum

Brief description of your project.

## Table of Contents
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)

## Installation

```bash
git clone https://github.com/VitasLu/EADSum.git
cd EADSum
conda env create -f environment.yml
conda activate EADSum
```

## Dataset Format

```bash
{
"id": 0,
"src": "By . Associated Press . PUBLISHED: . 14:11 EST, 25 October 2013 . | . UPDATED: . 15:36 EST, 25 October 2013 . The bishop of the Fargo Catholic Diocese in North Dakota has exposed potentially hundreds of church members in Fargo, Grand Forks and Jamestown to the hepatitis A virus in late September and early October. The state Health Department has issued an advisory of exposure for anyone who attended five churches and took communion. Bishop John Folda (pictured) of the Fargo Catholic Diocese in North Dakota has exposed potentially hundreds of church members in Fargo, Grand Forks and Jamestown to the hepatitis A . State Immunization Program Manager Molly Howell says the risk is low, but officials feel it's important to alert people to the possible exposure. The diocese announced on Monday that Bishop John Folda is taking time off after being diagnosed with hepatitis A. The diocese says he contracted the infection through contaminated food while attending a conference for newly ordained bishops in Italy last month. Symptoms of hepatitis A include fever, tiredness, loss of appetite, nausea and abdominal discomfort. Fargo Catholic Diocese in North Dakota (pictured) is where the bishop is located .",
"original_summary": "Bishop John Folda, of North Dakota, is taking time off after being diagnosed .He contracted the infection through contaminated food in Italy .Church members in Fargo, Grand Forks and Jamestown could have been exposed .",
"element-aware": "Important entities: Bishop John Folda, Fargo Catholic Diocese, North Dakota, state Health Department, State Immunization Program Manager Molly Howell.Important dates: Late September and early October, 2013.Events: Bishop John Folda contracted hepatitis A while attending a conference for newly ordained bishops in Italy. He then potentially exposed church members in Fargo, Grand Forks, and Jamestown to the virus through communion. The state Health Department issued an advisory of exposure and the diocese announced that Bishop Folda is taking time off.Result: Potentially hundreds of church members have been exposed to hepatitis A and Bishop Folda is taking time off due to his diagnosis. The state Health Department has issued an advisory to those who attended the affected churches."
},
```

## Training

To train the model, follow these steps:

1. Prepare your dataset:
   ```bash
   python data_utils.py --dataset cnndm
   ```

2. Start training:
   ```bash
   python DT/run.py --from_pretrained google-t5/t5-base --dataset cnndm --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 4
   ```

3. Standard Finetuning
   ```bash
   python DT/run.py --from_pretrained google-t5/t5-base --dataset cnndm --model_type standard --label_type gt --batch_size 4
   ```

## Inference

To run inference using a trained model:

```bash
python inference.py 
```
