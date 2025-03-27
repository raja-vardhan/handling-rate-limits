import com.google.common.base.Stopwatch;
import com.google.common.util.concurrent.RateLimiter;
import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.ModelType;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.errors.RateLimitException;
import com.openai.models.ChatModel;
import com.openai.models.responses.Response;
import com.openai.models.responses.ResponseCreateParams;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class Main
{
    public static void main(String[] args)
    {

        int totalNumOfRequests = 200;

        // Rate limiter for limiting requests to the model
        double tokensPerMinute = 30000;
        double requestsPerMinute = 500;

        RateLimiter tokenRateLimiter = RateLimiter.create(tokensPerMinute/60);
        RateLimiter requestRateLimiter = RateLimiter.create(requestsPerMinute/60);

        // The same query will be sent
        String query = "What is tokenization in large language models?";

        // Keep track of the total tokens sent
        AtomicInteger estimatedTotalTokens = new AtomicInteger(0);
        AtomicInteger actualTotalTokens = new AtomicInteger(0);
        AtomicInteger failedRequests = new AtomicInteger(0);
        AtomicInteger failedAttempts = new AtomicInteger(0);

        // Count the tokens for throttling requests
        EncodingRegistry encodingRegistry = Encodings.newDefaultEncodingRegistry();
        Encoding encoding = encodingRegistry.getEncodingForModel(ModelType.GPT_4O);

        OpenAIClient client = OpenAIOkHttpClient.builder()
                .apiKey("YOUR-API-KEY")
                .build();

        Stopwatch stopwatch = Stopwatch.createStarted();

        try (ExecutorService executorService = Executors.newFixedThreadPool(64)) {

            for (int i = 0; i < totalNumOfRequests; i++) {
                executorService.submit(() -> {

                    int maxTries = 2;
                    int backOffTime = 60;

                    ResponseCreateParams params = ResponseCreateParams.builder()
                            .input(query)
                            .model(ChatModel.GPT_4O)
                            .maxOutputTokens(400)
                            .build();

                    int estimatedTokensForRequest = encoding.countTokens(query) + 400;  // Approximation as the count of input tokens
                                                                                        // and max output tokens field in params

                    for (int currentTry = 1; currentTry <= maxTries; currentTry++) {

                        tokenRateLimiter.acquire(estimatedTokensForRequest);
                        requestRateLimiter.acquire(1);

                        try {

                            Stopwatch timeForResponse = Stopwatch.createStarted();
                            System.out.println("Sending request");
                            Response response = client.responses().create(params);
                            timeForResponse.stop();

                            int actualTokensForRequest = (int) response.usage().get().totalTokens();
                            estimatedTotalTokens.addAndGet(estimatedTokensForRequest);
                            actualTotalTokens.addAndGet(actualTokensForRequest);


                            response.output().stream()
                                    .flatMap(item -> item.message().stream())
                                    .flatMap(message -> message.content().stream())
                                    .flatMap(content -> content.outputText().stream())
                                    .forEach(outputText -> {
                                        String info = System.lineSeparator() + "<---------------------->" + System.lineSeparator() +
                                                "Response:" + System.lineSeparator() +
                                                outputText.text() + System.lineSeparator() + System.lineSeparator() +
                                                "Metrics:" + System.lineSeparator() +
                                                String.format("Estimated: %d vs actual: %d  %n", estimatedTokensForRequest, actualTokensForRequest) +
                                                String.format("Time for response: %d sec %n", timeForResponse.elapsed(TimeUnit.SECONDS)) +
                                                System.lineSeparator() + "<---------------------->";
                                        System.out.println(info);
                                    });

                            break;

                        } catch (RateLimitException e) {
                            System.out.println(e.getMessage());
                            failedAttempts.addAndGet(1);
                            if(currentTry == maxTries) {
                                failedRequests.addAndGet(1);
                                System.out.printf("Request failed %d times. Stopping retry%n", maxTries);
                            } else {
                                System.out.printf("Request failed %d times. Retrying in %d sec%n", currentTry, backOffTime);
                                try {
                                    Thread.sleep(backOffTime * 1000L);
                                    backOffTime *= 2;
                                }
                                catch (InterruptedException e1) {
                                    Thread.currentThread().interrupt();
                                }
                            }
                        } catch (Exception e) {
                            System.out.println(e.getMessage());
                            return;
                        }
                    }

                });
            }

            executorService.shutdown();
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        stopwatch.stop();
        long timeInSeconds = stopwatch.elapsed(TimeUnit.SECONDS);

        System.out.printf("Time elapsed %d min %d sec%n", timeInSeconds/60, timeInSeconds%60);

        System.out.println("Total estimated tokens for all requests: " + estimatedTotalTokens.get());
        System.out.println("Total actual tokens for all requests: " + actualTotalTokens.get());
        System.out.println("Failed attempts: " + failedAttempts.get());
        System.out.println("Failed requests: " + failedRequests.get());
    }
}
