package com.mycompany.revisao;

import java.lang.StringBuilder;
import java.util.ArrayList;

public class Palavra {

    private String palavra;
    private String[] dicas;
    private Rodada rodada;

    //construtor
    public Palavra(String palavra) {
        this.palavra = palavra.toUpperCase();

    }
    
     public Palavra(Palavra outra) {
        this.palavra = outra.palavra;
        

    }

    //getter
    public Rodada getRodada() {
        return rodada;
    }

    public String getPalavra() {
        return palavra;
    }

    public String[] getDicas() {
        return dicas;
    }

    //setters
    public void setRodada(Rodada rodada) {
        this.rodada = rodada;
    }

    public void setPalavra(String palavra) {
        this.palavra = palavra.toUpperCase(); 

    }

    public void setDicas(String[] dicas) {
        this.dicas = dicas;

    }

    //só verifica quantos caracteres iguais ao escolhido a palavra tem
    public int verificarCaracter(char c) {

        //divide a palavra em caracteres
        //compara em looping com o c e coloca dentro de um contador o valor
        int count = 0;

        for (char letra : palavra.toCharArray()) {
            if (letra == c) {
                count++;
            }
        }

        return count;
    }

    //será usado para controlar quantas sugestões o usuario vai usar - ver se é necessário (acho que nn)
    public int contasLetras() {

        int count = 0;

        for (int i = 0; i < palavra.length(); i++) {
            count++;
        }

        return count;
    }

    public String imprimirOriginal() {

        return palavra.toUpperCase();
    }

    //conferir
    //tinha que ser vetor porque assim só consegue usar uma letra, mas ok
    public String imprimirCriptografada(char[] letrasJogadas) {

        StringBuilder result = new StringBuilder();
        boolean encontrada = false;

    
        for (char letra : palavra.toCharArray()) {
            for(char jogado : letrasJogadas){

            if(Character.toUpperCase(jogado) == letra){
            
                encontrada = true;
           
                //operador ternario 
                //se encontrada for verdadeiro ele mostra encontrada : (eh como se fosse um else) else retorna um *
                result.append(encontrada ? letra : '*');
                
            }
            
        }

        return result.toString();

    }

    //metodo para dividir palavra em char
    public char[] palavraChar() {
        return palavra.toCharArray();

    }
}

