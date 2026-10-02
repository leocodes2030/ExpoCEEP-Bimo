import serial
import time

esp = serial.Serial("COM13", 115200)
time.sleep(2)

print("BIMO conectado!")
print("Digite uma mensagem para aparecer na tela.")
print("Digite 'sair' para fechar.\n")

while True:
    texto = input("Você: ")

    if texto.lower() == "sair":
        break

    esp.write((texto + "\n").encode("utf-8"))

esp.close()
print("BIMO desconectado.")
