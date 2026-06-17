package com.mycompany.revisao;

import java.util.ArrayList;
import java.util.Scanner;

public class Rodada {

    Scanner scanner = new Scanner(System.in);

    private Palavra palavra;
    private Jogador jogador;
    private int nJogadas;
    private ArrayList <Character> charJogados;
    private boolean isVencedor;
    private boolean isExecutando;
    private char palpite;
    private ArrayList<Character> charPalpites = new ArrayList<>(); 

//Método Construtor: 
    public Rodada(Palavra palavra, Jogador jogador) {
        this.palavra = palavra;
        this.jogador = jogador;
        this.nJogadas = 0;
        this.charJogados =;
        this.isVencedor = false;
        this.isExecutando = true;
        this.palpite = ' ';
        charPalpites = new ArrayList<>(); 
    }

//Getters e Setters
    public Palavra getPalavra() {
        return palavra;
    }

    public Jogador getJogador() {
        return jogador;
    }

    public int getNjogadas() {
        return nJogadas;
    }

    public char[] getCharJogados(){
return charJogados;
}
    public boolean getIsVencedor() {
        return isVencedor;
    }

    public boolean getIsExecutando() {
        return isExecutando;
    }

    ////////////////////
    /// @param palavra/

public void setPalavra(Palavra palavra) {
        this.palavra = palavra;
    }

    public void setJogador(Jogador jogador) {
        this.jogador = jogador;
    }

    public void setNjogadas(int nJogadas) {
        this.nJogadas = nJogadas;
    }

    public void setCharJogados(char[] charJogados){
    this.charJogados = charJogados;
}
    public void setIsVencedor(boolean isVencedor) {
        this.isVencedor = isVencedor;
    }

    public void setIsExecutando(boolean isExecutando) {
        this.isExecutando = isExecutando;
    }

    public void chamarClasseExterna(Palavra cripto, ArrayList<Character> charJogados) {
        //o metodo recebe de parametro um arraylist
        cripto.imprimirCriptografada(charJogados);
    }

    public void executarRodada() {
        int count = 0;
        int maxErros = 6;
        
        palavra.imprimirCriptografada(charPalpites);

                while(count < maxErros){
                   System.out.println("digite um palpite de letra: \n");
            palpite = scanner.next().charAt(0);
            charPalpites.add(palpite);

               if(!palavra.palavraChar().equals(palpite)){
               break;
               }
               
               count++;
        
        }

    }
      
    //retorna o número de erros - revisar
    public int qtdErros() {
        int count = 0;
//o contador aumenta de acordo com as comparações entre a palavra e o palpite - fiz um arraylist com todos os palpites
//vou comparar o arraylist com char palpites com palavraChar 
        for (char passa : palavra.palavraChar()) {
            if ((!charPalpites.equals(passa))) { //se os palpitesforem diferentes dos caracteres de palavra conta um
                count++;
            }
        }

        return count;
    }

//retorna o número de acertos.
    public int qtdAcertos() {
        int count = 0;

        for (char passa : palavra.palavraChar()) {
            if ((!charPalpites.equals(passa))) { //se os palpitesforem diferentes dos caracteres de palavra conta um
                count++;
            }
        }

        return count;
    }

    ///define vencedor
    public void defineVencedor() {

        isVencedor = true;

        if (qtdAcertos() != qtdErros()) {
            isVencedor = false;
        }
    }

   
    }

    

    //ok, eu quero definir o vencedor - ele é baseado em quem comete menos erros entre todo mundo que jogou

