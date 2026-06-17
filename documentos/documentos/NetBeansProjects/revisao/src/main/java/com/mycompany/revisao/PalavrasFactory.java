package com.mycompany.revisao;

import java.util.ArrayList;

public class PalavrasFactory {

    /* Método Estático:
○ criarBancoPalavras(): cria 20 objetos Palavra e retorna em um
ArrayList<Palavra>.*/
 //   private ArrayList<Palavra> listaPalavras = new ArrayList<>(); 

    
    public ArrayList<Palavra> criarBancoPalavras() { 

        ArrayList<Palavra> listaPalavras = new ArrayList();

       Palavra palavraUm = new Palavra("Planta");
        listaPalavras.add(palavraUm);

        Palavra palavraDois = new Palavra("Cadeira");
        listaPalavras.add(palavraDois);

        Palavra palavraTres = new Palavra("Televisao");
        listaPalavras.add(palavraTres);

        Palavra palavraQuatro = new Palavra("Casaco");
        listaPalavras.add(palavraQuatro);

        Palavra palavraCinco = new Palavra("Mesa");
        listaPalavras.add(palavraCinco);

        Palavra palavraSeis = new Palavra("Computador");
        listaPalavras.add(palavraSeis);

        Palavra palavraSete = new Palavra("Janela");
        listaPalavras.add(palavraSete);

        Palavra palavraOito = new Palavra("Livro");
        listaPalavras.add(palavraOito);

        Palavra palavraNove = new Palavra("Cachorro");
        listaPalavras.add(palavraNove);

        Palavra palavraDez = new Palavra("Gato");
        listaPalavras.add(palavraDez);

        Palavra palavraOnze = new Palavra("Relogio");
        listaPalavras.add(palavraOnze);

        Palavra palavraDoze = new Palavra("Chave");
        listaPalavras.add(palavraDoze);

        Palavra palavraTreze = new Palavra("Sapato");
        listaPalavras.add(palavraTreze);

        Palavra palavraQuatorze = new Palavra("Bicicleta");
        listaPalavras.add(palavraQuatorze);

        Palavra palavraQuinze = new Palavra("Escola");
        listaPalavras.add(palavraQuinze);

        Palavra palavraDezesseis = new Palavra("Celular");
        listaPalavras.add(palavraDezesseis);

        Palavra palavraDezessete = new Palavra("Bolsa");
        listaPalavras.add(palavraDezessete);

        Palavra palavraDezoito = new Palavra("Copo");
        listaPalavras.add(palavraDezoito);

        Palavra palavraDezenove = new Palavra("Porta");
        listaPalavras.add(palavraDezenove);

        Palavra palavraVinte = new Palavra("Carro");
        listaPalavras.add(palavraVinte);

        return listaPalavras;

    }

}
