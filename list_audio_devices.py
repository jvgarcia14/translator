import soundcard as sc
print("\nAvailable recording and loopback devices:\n")
for d in sc.all_microphones(include_loopback=True):
    print(d.name)
print("\nDefault speaker:\n" + sc.default_speaker().name)
input("\nPress Enter to close...")
