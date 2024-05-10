package org.chainoptimconfig;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.config.server.EnableConfigServer;

@SpringBootApplication
@EnableConfigServer
public class ChainOptimConfigServerApplication {

	public static void main(String[] args) {
		SpringApplication.run(ChainOptimConfigServerApplication.class, args);
	}

}
