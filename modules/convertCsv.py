import os, json
original = os.path.abspath(".")
if str(original).endswith('cs791'): root = original
else: root = os.path.abspath("../")
autoFolder = os.path.join(root, "AutoML", "ExpanseChameleon (askl2)")
traditionalFolder = os.path.join(root, "Traditional", "ExpanseChameleon")


try:
  for dir in [autoFolder]:
    os.chdir(dir)
    theseItems = os.listdir()
    for item in theseItems:
      if item.endswith('.json'):
      #if item == "Same Network With Transformation.json":
        with open(item, "r") as infile:
          tmp = json.loads(infile.read())
        
        try:
          newOut = "Transfer,CC,Model,Accuracy\n"
          for model in tmp:
            for cc in tmp[model]:
              for transfer in tmp[model][cc]:
                thisModel = model.replace(',',';').replace('\n', '').replace('\n', '')
                newOut += f"{transfer},{cc},{thisModel},{tmp[model][cc][transfer]['Average']}\n"
        except Exception as e:
          print(e)
          newOut = "File,CC,Model,Accuracy\n"
          for env in tmp:
            for cc in tmp[env]:
              for file in tmp[env][cc]:
                for model in tmp[env][cc][file]:
                  thisModel = model.replace(',',';').replace('\n', '').replace('\n', '')
                  try: newOut += f"{file},{cc},{thisModel},{tmp[env][cc][file][model]['Accuracy']}\n"
                  except: newOut += f"{file},{cc},{thisModel},{tmp[env][cc][file][model]}\n"
        
        with open(item.replace('.json', '.csv'), 'w') as f:
          f.write(newOut)
  os.chdir(original)
except Exception as e:
  print(e)
  os.chdir(original)