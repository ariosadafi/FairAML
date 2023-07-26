import os

experiments = ["0", "1", "2", "3", "4"]
folds = [0,1,2,3,4]
kind = ["sex", "age"]

if os.path.exists("run.sh"):
    os.remove("run.sh")

f = open("run.template", "r")
sh = open("run.sh", "w")

sh.writelines("#!/bin/bash\n\n")

lines = f.readlines()

for k in kind:
    for exp in experiments:
        for fld in folds:
            if os.path.exists("sb-" + k + "-" + exp + "-" + str(fld) + ".cmd"):
                os.remove("sb-" + k + "-" + exp + "-" + str(fld) + ".cmd")


for k in kind:
    for exp in experiments:
        for fld in folds:
                c = lines.copy()
                for i in range(len(c)):
                    c[i] = c[i].replace("@fld@", str(fld))
                    c[i] = c[i].replace("@exp@", str(exp))
                    c[i] = c[i].replace("@kind@", str(k))

                with open("sb-" + k + "-" + exp + "-" + str(fld) + ".cmd", "w") as g:
                    g.writelines(c)
                    g.flush()
                    g.close()

                sh.writelines("sb-" + k + "-" + exp + "-" + str(fld) + ".cmd\n")
            # sh.writelines("sleep 0.5\n")

sh.flush()
sh.close()


