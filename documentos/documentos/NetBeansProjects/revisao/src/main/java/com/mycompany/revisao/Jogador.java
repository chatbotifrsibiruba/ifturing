package com.mycompany.revisao;

public class Jogador {

    private String nome;
    private int totalVitorias;

    // Método Construtor: inicializa o nome do jogador e zera o total de vitórias.
    public Jogador(String nome) {
        this.nome = nome;
        this.totalVitorias = 0; //por zerar total de vitorias entendo como inicializar

    }

//○ Getters e Setters.
    public String getNome() {
        return nome;
    }

    public int getTotalVitorias() {
        return totalVitorias;
    }

    public void setNome(String nome) {
        this.nome = nome;

    }

    //sempre q eu acertar uma palavra chama esse metodo
   
    public void setTotalVitorias(int totalVitorias) {
        this.totalVitorias++;
    }

}
