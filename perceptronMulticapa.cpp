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
float biasOculto[NEU_OCULTAS];
float biasSalida[NEU_SALIDA];
float oculto[NEU_OCULTAS];
float salida[NEU_SALIDA];

void imprimeSalida(float* sal){
	for(int i = 0; i<NEU_SALIDA; i++){
		cout << sal[i] << endl;
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

void initBias(){
	int i;
	for(i = 0; i<NEU_OCULTAS; i++){
		biasOculto[i] = 1.0;
	}
	for(i = 0; i<NEU_SALIDA; i++){
		biasSalida[i] = 1.0;
	}
}

float sigmoide( float s ){
	return (1/(1+exp(-s)));
}

float* run( float* entrada ){
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
		oculto[j]-=biasOculto[i];
		oculto[j] = sigmoide(oculto[j]);
	}
	
	for(i = 0; i<NEU_OCULTAS; i++){
		for(j = 0; j<NEU_SALIDA; j++){
			salida[j]=oculto[i]*peso2[i][j];
		}
	}
	for(j = 0; j<NEU_SALIDA; j++){
		salida[j]-=biasSalida[j];
		salida[j] = sigmoide(salida[j]);
	}
	return salida;
}

void entrenaRed( float* entrada, float* salida ){
	int i, j;
	float* salidaObtenida;
	float errorSalida[NEU_SALIDA];
	float errorOculto[NEU_OCULTAS];

	salidaObtenida = run(entrada);

	//Calculamos los errores en la salida y modificamos el bias de cada 
	// neurona de salida.
	for(i = 0; i<NEU_SALIDA; i++){
		errorSalida[i] = salida[i] - salidaObtenida[i];
		biasSalida[i]-= K*errorSalida[i];
	}

	//Modificamos los pesos de las conexiones entre las neuronas ocultas y
	// las salidas en funcion del error.
	for(i = 0; i<NEU_OCULTAS; i++){
		for(j = 0; j<NEU_SALIDA; j++){
			peso2[i][j] += K*errorSalida[j]*oculto[i];
		}
	}

	//Inicializamos el error a cero.
	for(i=0; i<NEU_OCULTAS; i++){
		errorOculto[i]=0;
	}

	//Propagamos el error.
	for(i = 0; i<NEU_OCULTAS; i++){
		for(j = 0; j<NEU_SALIDA; j++){
			errorOculto[i]+=errorSalida[j] * peso2[i][j];
		}
	}
	
	//Modificamos el bias de las neuronas ocultas.
	for(i = 0; i<NEU_OCULTAS; i++){
		biasOculto[i]-= K*errorOculto[i];
	}
	
	//Modificamos los pesos de las conexiones entre las neuronas de entrada
	// y las ocultas.
	for(i = 0; i<NEU_ENTRADA; i++){
		for(j = 0; j<NEU_OCULTAS; j++){
			peso1[i][j] += K*errorOculto[j]*entrada[i];
		}
	}
}


int main(){
	float a[2] = {0,0};
	float b[2] = {0,1};
	float c[2] = {1,0};
	float d[2] = {1,1};
	float sa[2] = {0};
	float sb[2] = {1};
	float sc[2] = {1};
	float sd[2] = {1};
	initPesos();
	initBias();
	for(int i = 0; i<2000; i++){
		entrenaRed(a,sa);
		entrenaRed(b,sb);
		entrenaRed(c,sc);
		entrenaRed(d,sd);
	}
	float* sal;	
	sal = run(a);
	cout << "0,0" << endl;
	imprimeSalida(sal);
	
	sal = run(b);
	cout << "0,1" << endl;
	imprimeSalida(sal);
	
	sal = run(c);
	cout << "1,0" << endl;
	imprimeSalida(sal);
	
	sal = run(d);
	cout << "1,1" << endl;
	imprimeSalida(sal);
}


