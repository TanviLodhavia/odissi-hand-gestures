import os

path_to_model = "model/first_model.keras"
path_to_dataset = r"training" # training folder path

# mudra_names = ['Alapadma','Arala','Ardhachandra','Brahmara','Chandrakala','Chatura','Hamsapaksha','Hamsasya','Kangula','Kapitha',
#                'Karatarimukha','Katkamukha','Mayura','Mrigasisya','Mukula','Mushti','Padmakosha','Pataka','Samdamsha','Sarpashisha',
#                'Shikhara','Shukatunda','Simhamukha','Suchi','Tamarachuda','Tripataka','Trisula'
#                ]

mudra_names=['Alapadma', 'Chandrakala', 'Chatura', 'Hamsasya', 'Mushti', 'Padmakosha', 'Pataka', 'Suchi', 'Tamarachuda', 'Trisula']

input_shape = (224, 224)