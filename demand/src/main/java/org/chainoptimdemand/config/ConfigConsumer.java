package org.chainoptimdemand.config;

import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.stereotype.Component;

@Getter
@Component
@RefreshScope
public class ConfigConsumer {

    @Value("${test.value}")
    private String testValue;

}
