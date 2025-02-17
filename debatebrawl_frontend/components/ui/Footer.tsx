import { Box, Container, Text, Link, Flex } from '@chakra-ui/react';

export default function Footer() {
  return (
    <Box bg="gray.900" color="white" py={4}>
      <Container maxW="container.xl">
        <Flex justifyContent="space-between" alignItems="center" flexWrap="wrap">
          <Text>© 2025 DebateBrawl. All rights reserved</Text>
          <Flex alignItems="center">
            <Text mr={2}>Developed with love ❤️ by</Text>
            <Link href="https://prakasharyan.com/" isExternal color="blue.400" fontWeight="bold">
              Prakash Aryan
            </Link>
          </Flex>
        </Flex>
      </Container>
    </Box>
  );
}