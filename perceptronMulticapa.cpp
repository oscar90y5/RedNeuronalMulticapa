//Perceptron multicapa (sin retroalimentacion).
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>

using namespace std;

#define NEU_ENTRADA 2  //Numero de neuronas de entrada.
#define NEU_OCULTAS 4  //Numero de neuronas ocultas.
#define NEU_SALIDA 1   //Numero de neuronas de salida.
#define EPOCA 500      //Numero de iteraciones del entrenamiento.
#define K 0.2f         //Taza de aprendizaje del perceptron.

//Matrices que almacenan los pesos
float peso1[NEU_ENTRADA][NEU_OCULTAS]; //pesos entre neuronas de entrada y ocultas.
float peso2[NEU_OCULTAS][NEU_SALIDA]; //pesos entre neuronas ocultas y de salida.
float bias = 1.0f; //El bias lo ajustamos al valor 1
float oculto[NEU_OCULTAS];
float salida[NEU_SALIDA];

void imprimeSalida(){
	for(int i = 0; i<NEU_SALIDA; i++){
		cout << salida[i] << endl;
	}
}

void initPesos(){
	int i, j;
	srand48(time(NULL));

	for(i = 0; i<NEU_ENTRADA; i++){
		for(j = 0; j<NEU_OCULTAS; j++){
			peso1[i][j] = drand48();
		}
	}

	for(i = 0; i<NEU_OCULTAS; i++){
		for(j = 0; j<NEU_SALIDA; j++){
			peso2[i][j] = drand48();
		}
	}
}

float sigmoide( float s ){
	return (1/(1+exp(-s)));
}

void entrenaRed( float entrada[NEU_ENTRADA], float salida ){

}

float* run( float entrada[NEU_ENTRADA] ){
	int i, j;

	for(i=0;i<NEU_OCULTAS;i++){
		oculto[i]=0.0;
	}
	for(i=0;i<NEU_SALIDA;i++){
		salida[i]=0.0;
	}

	for(i = 0; i<NEU_ENTRADA; i++){
		for(j = 0; j<NEU_OCULTAS; j++){
			oculto[j]=entrada[i]*peso1[i][j];
		}
	}
	for(j = 0; j<NEU_OCULTAS; j++){
		oculto[j]-=bias;
		oculto[j] = sigmoide(oculto[j]);
	}
	
	for(i = 0; i<NEU_OCULTAS; i++){
		for(j = 0; j<NEU_SALIDA; j++){
			salida[j]=oculto[i]*peso2[i][j];
		}
	}
	for(j = 0; j<NEU_SALIDA; j++){
		salida[j]-=bias;
		salida[j] = sigmoide(salida[j]);
	}
	return salida;
}

int main(){
	float entrada[2] = {0,1};
	initPesos();
	float* salida;
	salida = run(entrada);
	imprimeSalida();
}

