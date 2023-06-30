## EMDA study

Empirical study of domain adaptation models. Follow these steps to run the methods.

### The repository contains the code for the following adaptation methods.

- [x] Plain
- [x] [DANN](https://github.com/fungtion/DANN)
- [x] [CDAN](https://github.com/thuml/CDAN)
- [x] [SAFN](https://github.com/jihanyang/AFN)
- [x] [MCC](https://github.com/thuml/Versatile-Domain-Adaptation)
- [x] [MemSAC](https://github.com/ViLab-UCSD/MemSAC_ECCV2022)
- [x] [MDD](https://github.com/thuml/MDD)
- [x] [MCD](https://github.com/mil-tokyo/MCD_DA)
- [x] [ToAlign](https://github.com/microsoft/UDA)
- [ ] [CGDM](https://github.com/lijin118/CGDM/blob/main/train_domainnet.py)
- [ ] [FixBi](https://github.com/NaJaeMin92/FixBi/blob/main/trainer/fixbi_trainer.py)

### Use the following command to run the training of models.

If you have to run on CompCars, for example

#### Plain
```
python3 train.py --config configs/plain.yml --source data/compcars/night_srv_train.txt --target data/compcars/day_srv_train.txt --num_class 281 --data_root /newfoundland/tarun/datasets/Adaptation/CompCars/srv/sv_data --num_iter 60000 --exp_name cars/plain_night_day --trainer plain
```

#### CDAN
```
python3 train.py --config configs/cdan.yml --source data/compcars/night_srv_train.txt --target data/compcars/day_srv_train.txt --num_class 281 --data_root /newfoundland/tarun/datasets/Adaptation/CompCars/srv/sv_data --num_iter 60000 --exp_name cars/cdan_night_day --trainer cdan
```

#### MemSAC
```
python3 train.py --config configs/memsac.yml --source data/compcars/night_srv_train.txt --target data/compcars/day_srv_train.txt --num_class 281 --data_root /newfoundland/tarun/datasets/Adaptation/CompCars/srv/sv_data --num_iter 60000 --exp_name cars/memsac_night_day --trainer memsac
```

Other methods and datasets are similar, just change the corresponding datafiles and trainers.
# GeoNet
# GeoNet
