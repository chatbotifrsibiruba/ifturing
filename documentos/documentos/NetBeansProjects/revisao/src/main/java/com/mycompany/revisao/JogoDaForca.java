package com.mycompany.revisao;

import java.util.ArrayList;
import java.util.Scanner;

public class JogoDaForca {

    Scanner scanner = new Scanner(System.in);

    String guardaPlayer = " ";
    int guardaRodadas = 0;

    private ArrayList<Rodada> listaRodadas; 
    private ArrayList<Jogador> listaJogadores; 
    private ArrayList<Palavra> listaPalavras; 

    PalavrasFactory palavrasFactory = new PalavrasFactory();

    //metodo construtor
    public JogoDaForca() {
        this.listaPalavras = palavrasFactory.criarBancoPalavras();

    }

//getters e setters
    public ArrayList<Rodada> getListaRodadas() {
        return listaRodadas;
    }

    public ArrayList<Jogador> getListaJogadores() {
        return listaJogadores;
    }

    public ArrayList<Palavra> getListaPalavras() {
        return listaPalavras;
    }

    ////////////////
    /// @param listaRodadas/

public void setListaRodadas(ArrayList<Rodada> listaRodadas) {
        this.listaRodadas = listaRodadas;
    }

    public void setListaJogadores(ArrayList<Jogador> listaJogadores) {
        this.listaJogadores = listaJogadores;
    }

    public void setListaPalavras(ArrayList<Palavra> listaPalavras) {
        this.listaPalavras = listaPalavras;
    }

    public void configurarJogo() { //não pode ter parametro
        
        System.out.println("Configuração do jogo \n");
        System.out.println("Digite a lista de nomes, para parar, clique 0");

        
        while (true){
            System.out.println("Qual é o seu nome? ");
            guardaPlayer = scanner.next(); 
            
            if(guardaPlayer.equals(0)){
                System.out.println("fim da lista de nomes ");
                break;
            }
            
            Jogador player = new Jogador(guardaPlayer);
            listaJogadores.add(player);

        } 
        
        System.out.println("Jogadores cadastrados: ");
        for(Jogador c: listaJogadores){
            System.out.println(c);
        }
        
        
        System.out.println("Configuração de quantidade de rodadas ");

        System.out.println("Qual é o número de rodadas por jogador?");
        guardaRodadas = scanner.nextInt();
        //sem loop pq vale pra todos
        
        
        
        //cria Rodada e add em listaRodadas
        int totalRodadas = listaJogadores.size() * guardaRodadas;
        
       for (int i = 0; i < guardaRodadas; i++) {

        for (Jogador player : listaJogadores) {
                            
            Palavra palavra = listaPalavras.get(i % listaPalavras.size()); //CHAT GPT

            Rodada rodada = new Rodada(palavra, player); //cada rodada é um jogador e uma palavra 

            listaRodadas.add(rodada);
            }
        }
    }

    //terminar
    public void iniciarJogo() {
        //parametro ArrayList<Character> listaCaractere porque para inicializar o jogo precisa de uma lista de caracteres

        //pede o palpite 
        for (Rodada passa : listaRodadas) {
            passa.executarRodada();
        }

    }

    public void finalizarJogo() {

        for (Rodada passa : listaRodadas) {
            passa.qtdAcertos();
            passa.qtdErros();
            passa.defineVencedor();
            //como exibir o placar? os nomes estão dentro da listaJogadores eu quero mostrar do melhor jogador ao pior
            //

        }

    }

}
