import os
import openai

#import para las colas
import queue

#reconocimiento facial
import cv2
from fer import FER
import time

#import de variables de entorno
from dotenv import load_dotenv

#import para multihilos
import threading
from multiprocessing import Process

#import para text to speech y velocidades
from gtts import gTTS
from pydub import AudioSegment


#imports para grabar la voz
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

#deteccion de silencio
import wave
import pygame

#imports para transformar un audio a texto almacenable
import speech_recognition as sr

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()

#clave openai
#openai.api_key = 'AQUI_TU_CLAVE_API_OPENAI'
openai.api_key = os.getenv('OPENAI_API_KEY') #o usala como variable de entorno


#carga el modelo y el tokenizador
model_name = "microsoft/DialoGPT-medium"  #Puedes usar "DialoGPT-small", "DialoGPT-medium", o "DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#mueve el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pygame.mixer.init() #Inicia el mixer

#inicia el detector de emociones
detector = FER()

#crea una cola para pasar emociones entre hilos
q = queue.Queue()

def detectarEmociones(q):

    #inicia el contador de emociones a 0
    total_enfado = 0
    total_feliz = 0
    total_triste = 0
    total_miedo = 0
    total_sorpresa = 0
    total_neutro = 0
    total_disgusto = 0

    #captura desde la webcam
    cap = cv2.VideoCapture(0)

    #tiempo inicial
    last_print_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            #detecta las emociones en el frame
            emotions = detector.detect_emotions(frame)

            #procesa las emociones detectadas
            if isinstance(emotions, list):  #verifica si 'emotions' es una lista
                for emotion in emotions:    #para cada emocion en la lista de emociones
                    if isinstance(emotion, dict) and 'box' in emotion and 'emotions' in emotion:
                        (x, y, w, h) = emotion['box']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)    #reconocimiento de emociones

                        #usa directamente las emociones sin llamar a 'top_emotion' y suma el porcentual de cada emocion
                        emotion_dict = emotion['emotions']
                        total_enfado += emotion_dict.get('angry', 0)
                        total_feliz += emotion_dict.get('happy', 0)
                        total_triste += emotion_dict.get('sad', 0)
                        total_miedo += emotion_dict.get('fear', 0)
                        total_sorpresa += emotion_dict.get('surprise', 0)
                        total_neutro += emotion_dict.get('neutral', 0)
                        total_disgusto += emotion_dict.get('disgust', 0)

            #calcula el tiempo actual
            current_time = time.time()
            
            #si el tiempo es menor que 0.5 y almacena en una lista todos los valores de sentimiento menos el que se va a comparar
            if current_time - last_print_time >= 0.5:
                valoresEnfado = [total_disgusto,total_feliz,total_miedo,total_neutro,total_sorpresa,total_triste]
                valoresFeliz = [total_disgusto,total_enfado,total_miedo,total_neutro,total_sorpresa,total_triste]
                valoresTriste = [total_disgusto,total_feliz,total_miedo,total_neutro,total_sorpresa,total_enfado]
                valoresMiedo = [total_disgusto,total_feliz,total_enfado,total_neutro,total_sorpresa,total_triste]
                valoresSorpresa = [total_disgusto,total_feliz,total_miedo,total_neutro,total_enfado,total_triste]
                valoresNeutro = [total_disgusto,total_feliz,total_miedo,total_enfado,total_sorpresa,total_triste]
                valoresDisgusto = [total_enfado,total_feliz,total_miedo,total_neutro,total_sorpresa,total_triste]
                #si el sentimiento es mayor que cualquiera de la lista ese sentimiento es el actual del usuario
                if (total_enfado > max(valoresEnfado)):
                    print("El usuario está enfadado")
                    emocion_detectada = 'enfado'
                if (total_feliz > max(valoresFeliz)):
                    print("El usuario está feliz")
                    emocion_detectada = 'felicidad'
                if (total_triste > max(valoresTriste)):
                    print("El usuario está triste")
                    emocion_detectada = 'tristeza'
                if (total_miedo > max(valoresMiedo)):
                    print("El usuario tiene miedo")
                    emocion_detectada = 'miedo'
                if (total_sorpresa > max(valoresSorpresa)):
                    print("El usuario ha sido sorprendido")
                    emocion_detectada = 'sorpresa'
                if (total_neutro > max(valoresNeutro)):
                    print("El usuario está neutro")
                    emocion_detectada = 'neutralidad'
                if (total_disgusto > max(valoresDisgusto)):
                    print("El usuario está disgustado")
                    emocion_detectada = 'disgusto'
                
                #si hay una emocion
                if emocion_detectada:
                    with q.mutex:
                        q.queue.clear()  #limpia la cola
                    q.put_nowait(emocion_detectada)  #encola la última emoción detectada
                else:
                    emocion_detectada = 'neutralidad'
                #reinicia las emociones (SE PUEDE MEJORAR ESTE ALGORITMO HACIENDO UNA MEDIA PONDERADA O UNA SUMA DE LOS TOTALES PARA VER CUAL PREVALECE EN GENERAL Y NO CADA ITERACION)
                total_disgusto = 0
                total_enfado = 0
                total_sorpresa = 0
                total_miedo = 0
                total_feliz = 0
                total_triste = 0
                total_neutro = 0
                last_print_time = current_time  #actualiza el tiempo

        except Exception as e:
            print('No se detecta emocion en el frame, es tomara en cuenta neutralidad')
            emocion_detectada = 'neutralidad'
            #nos aseguramos que haya acceso exclusivo
            with q.mutex:
                q.queue.clear() #Limpiamos la cola
            #agregamos la neutralidad a la cola
            q.put_nowait(emocion_detectada)

        cv2.imshow('Emotion Detection', frame)

        #se acaba la ejecucion de la funcion si se da a la Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def grabar_audio_dinamico(nombre_archivo, umbral_silencio=1.0, duracion_buffer=0.5, max_duracion=30):
    
    frecuencia_muestreo = 16000  #frecuencia de muestreo (Hz)
    print("Comenzando grabación...")


    pygame.mixer.music.set_volume(1.0)  #ajusta el volumen (0.0 a 1.0)
    pygame.mixer.music.load('C:/Users/Dary/Desktop/Proyectos/startRec.wav') #carga el sonido
    pygame.mixer.music.play()   #ejecuta el sonido

    buffers = []  #lista para almacenar los buffers de audio grabados
    silencio_detectado = False
    tiempo_silencio = 0  #contador de cuánto tiempo ha habido silencio
    duracion_total = 0   #contador de la duración total de la grabación

    while not silencio_detectado and duracion_total < max_duracion:
        #graba un buffer de duracion_buffer segundos (0.5)
        buffer = sd.rec(int(duracion_buffer * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='float32')
        sd.wait()

        #calcula el nivel de energía (potencia del sonido) del buffer grabado
        energia_buffer = np.linalg.norm(buffer)

        #cgrega el buffer grabado a la lista de buffers
        buffers.append(buffer)

        if energia_buffer < umbral_silencio:
            tiempo_silencio += duracion_buffer  #si el nivel de energía es bajo, aumentamos el contador de silencio
        else:
            tiempo_silencio = 0  #Si hay voz, reiniciamos el contador de silencio

        #si el silencio ha durado al menos 2 segundos consecutivos, paramos la grabación
        if tiempo_silencio >= 2:
            #print("Silencio detectado, terminando grabación...")
            silencio_detectado = True

        duracion_total += duracion_buffer
    
    pygame.mixer.music.load('C:/Users/Dary/Desktop/Proyectos/endRec.wav') #carga el sonido
    pygame.mixer.music.play()   #ejecuta el sonido
    #print("Guardando archivo...")
    
    #unimos todos los buffers en un solo array
    audio_total = np.concatenate(buffers)

    #guardar el archivo WAV
    audio_total = np.int16(audio_total * 32767)  #convertir a 16 bits
    with wave.open(nombre_archivo, 'w') as archivo_wav:
        archivo_wav.setnchannels(1)  #Mono
        archivo_wav.setsampwidth(2)  #16 bits
        archivo_wav.setframerate(frecuencia_muestreo)
        archivo_wav.writeframes(audio_total.tobytes())

    #print(f"Archivo guardado como: {nombre_archivo}")

#funcion para transformar el audio de un archivo a texto en una variable
def audio_a_texto(nombre_archivo):
    reconocedor = sr.Recognizer()
    
    with sr.AudioFile(nombre_archivo) as source:
        audio = reconocedor.record(source)  #lee el archivo de audio
    
    try:
        texto = reconocedor.recognize_google(audio, language="es-ES")  #puedes cambiar a 'en-US' para inglés
        #print(f"Texto detectado: {texto}")
        return texto
    except sr.UnknownValueError:
        print("No se pudo entender el audio")
        return ''
    except sr.RequestError as e:
        print(f"Error al solicitar el servicio de reconocimiento: {e}")

def generadorDeRespuesta(q):

    #no se necesita coma porque se recibe en forma de tupla
    model="gpt-4"  #gpt-3.5-turbo o gpt-4
    messages=[
            {"role": "system", "content": "Eres un asistente cuyo objetivo es conversar con un usuario basado en sus emociones. Primero recibirás la emoción en forma de emoción detecectada y luego su mensahe. Ten en cuenta de que deberás responder en consecuencia a sus emociones para lograr conectar más él. No hace falta que digas qué emoción ves, a no ser que te pregunte, entonces deberás responder que emoción recibiste. Si el usuario está triste intenta hablar de forma delicada, si el usuario está feliz comparte su alegría, si el usuario está enfadado comparte su frustración, y así con el resto de emociones... No hace falta que seas demasiado explícito, tan solo utiliza un lenguaje de acuerdo a sus emociones"}
        ]
    
    #se realiza mientras que se entienda y no haya silencios
    
    while True:
        grabar_audio_dinamico('prompt.wav')  #graba el audio hasta callarse
        texto = audio_a_texto('prompt.wav')  #transforma el audio a texto

        if texto == '':
            print('No se ha detectado ninguna entrada')
            break
        
        #extrae la emoción de la cola (si está disponible)
        try:
            emocion = q.get_nowait()  #intenta obtener la emoción de la cola sin esperar
        #si está vacía utilizará la neutralidad
        except queue.Empty:
            emocion = 'neutralidad'  #si no hay emoción disponible, usar neutralidad
        
        #agrega la respuesta al historial de mensajes
        messages.append({"role": "user", "content": f'Emoción detectada: {emocion}. Mensaje: {texto}'})
        
        #muestra el mensaje
        print(messages)
        
        #genera la respuesta
        response = openai.chat.completions.create(
        model=model,
        messages=messages
        )
        
        content = response.choices[0].message.content
        print(content)
        
        #generamos el text to speech
        tts = gTTS(text=content, lang='es')
        tts.save("output.mp3")  
        
        #cambia la velocidad del audio usando pydub
        audio = AudioSegment.from_file("output.mp3")
        audio_faster = audio.speedup(playback_speed=1.3)  #ajusta la velocidad (1.5x más rápido)
        audio_faster.export("output_faster.mp3", format="mp3")  #guarda el archivo ajustado
        
        #reproducimos el audio
        pygame.mixer.music.load('C:/Users/Dary/Desktop/Proyectos/output_faster.mp3') #carga el sonido
        pygame.mixer.music.play()   #ejecuta el sonido
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        #agrega la respuesta del asistente al historial de la conversación
        messages.append({"role": "assistant", "content": content})
        
        #si el mensaje del usuario es adiós se acaba el programa
        if texto.lower() == 'adiós':
            break


#ejecución de los hilos
if __name__ == '__main__':
    #crea un hilo para la detección de emociones
    hilo_emocion = threading.Thread(target=detectarEmociones, args=(q,))
    hilo_emocion.start()

    #ejecuta el generador de respuestas en el hilo principal
    generadorDeRespuesta(q)

    #asegura de que el hilo de emociones termine correctamente
    hilo_emocion.join()