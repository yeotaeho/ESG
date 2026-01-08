package kr.yeotaeho.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication
@EnableDiscoveryClient
public class EsgApplication {

	public static void main(String[] args) {
		SpringApplication.run(EsgApplication.class, args);
	}

}
